#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d RSI Divergence for mean reversion in range markets.
# Uses Bollinger Band Width percentile to detect range (CHOP > 61.8) and trend (CHOP < 38.2) regimes.
# In range: fade extreme RSI(14) from 1d timeframe with divergence confirmation.
# In trend: follow 6h EMA(21) pullbacks to EMA(50) with volume confirmation.
# 1d RSI provides higher timeframe momentum context to avoid counter-trend traps.
# Designed for low trade frequency (12-25/year) with discrete sizing 0.25 to manage drawdown.
# Works in bull/bear by adapting to regime: mean revert in range, trend follow in trend.

name = "6h_BBWRegime_RSIDivergence_1dContext_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for RSI and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = pd.Series(df_1d['close'])
    delta_1d = close_1d.diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = (-delta_1d).where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.values
    
    # Align 1d RSI to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate Bollinger Bands on 6h for regime detection
    # BB(20,2) - 20 period, 2 std dev
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std() * 2
    upper_bb = basis + dev
    lower_bb = basis - dev
    bb_width = (upper_bb - lower_bb) / basis * 100  # Percent
    
    # BB Width percentile lookback 50 periods (~6-7 days of 6h data)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Regime thresholds
    chop_threshold_high = 61.8  # Range when BB Width percentile > 61.8 (low volatility)
    chop_threshold_low = 38.2   # Trend when BB Width percentile < 38.2 (high volatility)
    
    # 6h EMAs for trend following
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for BB width percentile and EMAs
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_rsi_1d = rsi_1d_aligned[i]
        curr_bb_width_percentile = bb_width_percentile[i]
        curr_ema_21 = ema_21[i]
        curr_ema_50 = ema_50[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma[i]
        
        # Determine regime
        is_range = curr_bb_width_percentile > chop_threshold_high
        is_trend = curr_bb_width_percentile < chop_threshold_low
        is_neutral = not (is_range or is_trend)
        
        if position == 0:  # Flat - look for new entries
            if is_range:
                # Range regime: mean reversion at RSI extremes with divergence
                # Long: RSI < 30 (oversold) AND price near lower BB (within 0.5*)
                # Short: RSI > 70 (overbought) AND price near upper BB (within 0.5*)
                bb_position = (curr_close - lower_bb[i]) / (upper_bb[i] - lower_bb[i]) if (upper_bb[i] - lower_bb[i]) > 0 else 0.5
                if (curr_rsi_1d < 30 and bb_position < 0.3):  # Oversold and near lower BB
                    signals[i] = 0.25
                    position = 1
                elif (curr_rsi_1d > 70 and bb_position > 0.7):  # Overbought and near upper BB
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_trend:
                # Trend regime: follow 6h EMA(21) pullbacks to EMA(50) with volume
                # Long: price > EMA21 > EMA50 AND pullback to EMA21 with volume
                # Short: price < EMA21 < EMA50 AND pullback to EMA21 with volume
                if (curr_close > curr_ema_21 and curr_ema_21 > curr_ema_50 and
                    curr_close <= curr_ema_21 * 1.005 and  # Within 0.5% of EMA21
                    curr_volume > curr_vol_ma * 1.2):  # Volume confirmation
                    signals[i] = 0.25
                    position = 1
                elif (curr_close < curr_ema_21 and curr_ema_21 < curr_ema_50 and
                      curr_close >= curr_ema_21 * 0.995 and  # Within 0.5% of EMA21
                      curr_volume > curr_vol_ma * 1.2):  # Volume confirmation
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral regime: no clear signal
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_range:
                # Exit range long: RSI > 50 (mean reversion complete) OR price > upper BB
                if curr_rsi_1d > 50 or bb_position > 0.95:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_trend:
                # Exit trend long: EMA21 < EMA50 (trend change) OR price < EMA50
                if curr_ema_21 < curr_ema_50 or curr_close < curr_ema_50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Neutral: exit on mean reversion
                if 40 < curr_rsi_1d < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_range:
                # Exit range short: RSI < 50 (mean reversion complete) OR price < lower BB
                if curr_rsi_1d < 50 or bb_position < 0.05:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_trend:
                # Exit trend short: EMA21 > EMA50 (trend change) OR price > EMA50
                if curr_ema_21 > curr_ema_50 or curr_close > curr_ema_50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Neutral: exit on mean reversion
                if 40 < curr_rsi_1d < 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals