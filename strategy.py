#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume regime filter.
# Uses 4h EMA50 for trend direction (bull/bear filter) and 1d volume percentile
# to identify high/low volatility regimes. In high volatility regimes (top 30%),
# trade momentum (RSI > 60 for long, < 40 for short). In low volatility regimes
# (bottom 30%), trade mean reversion at Bollinger Bands (2, 20). Volume filter
# avoids choppy markets. Designed for 15-30 trades/year with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume percentile rank (using 50-day lookback)
    vol_series = pd.Series(volume_1d)
    vol_rank = vol_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan,
        raw=False
    ).values
    vol_rank_aligned = align_htf_to_ltf(prices, df_1d, vol_rank)
    
    # Calculate 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_rank_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_4h_val = ema_4h_aligned[i]
        vol_rank_val = vol_rank_aligned[i]
        rsi_val = rsi[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        
        # Trend filter: price above/below 4h EMA50
        uptrend = price > ema_4h_val
        downtrend = price < ema_4h_val
        
        # Volume regime: high volatility (top 30%) or low volatility (bottom 30%)
        high_vol = vol_rank_val > 0.7
        low_vol = vol_rank_val < 0.3
        
        if position == 0:
            # Entry logic
            if high_vol:
                # High volatility: momentum trading
                if uptrend and rsi_val > 60:
                    signals[i] = 0.20
                    position = 1
                elif downtrend and rsi_val < 40:
                    signals[i] = -0.20
                    position = -1
            elif low_vol:
                # Low volatility: mean reversion
                if price <= bb_low and rsi_val < 40:
                    signals[i] = 0.20
                    position = 1
                elif price >= bb_up and rsi_val > 60:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on RSI reversal or mean reversion signal
                if rsi_val < 40 or price >= bb_up:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on RSI reversal or mean reversion signal
                if rsi_val > 60 or price <= bb_low:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA_VolRegime_MomentumMeanRev"
timeframe = "1h"
leverage = 1.0