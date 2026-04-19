#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w ADX trend filter and 1d price action
# - 1w ADX > 25 indicates trending market, < 20 indicates ranging
# - In trending markets (ADX > 25): buy pullbacks to 20 EMA in uptrend, sell rallies to 20 EMA in downtrend
# - In ranging markets (ADX < 20): fade extreme RSI readings with Bollinger Band boundaries
# - Volume confirmation: require volume > 1.5x 20-day average for conviction
# - Position size: 0.25 (25%) to balance opportunity and risk
# - Designed to work in both bull (trending) and bear (ranging) markets by adapting to regime
# - Target: 15-25 trades/year to minimize fee drag while capturing meaningful moves

name = "1d_ADX_Regime_ADAPTIVE_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w ADX(14) for trend strength
    # Calculate True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift())
    tr3 = abs(df_1w['low'] - df_1w['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df_1w['high'] - df_1w['high'].shift()
    down_move = df_1w['low'].shift() - df_1w['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_14
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    
    # Align 1w ADX to daily
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # Get 1d data for entry signals
    # 20-period EMA for trend following
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Bollinger Bands(20,2) for mean reversion boundaries
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        adx_val = adx_1w_aligned[i]
        
        if position == 0:
            # Look for entries based on regime
            if adx_val > 25:  # Trending regime
                # In uptrend: buy dip to EMA20 with volume
                if close[i] > ema_20[i] and close[i] <= ema_20[i] * 1.02 and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # In downtrend: sell rally to EMA20 with volume
                elif close[i] < ema_20[i] and close[i] >= ema_20[i] * 0.98 and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (ADX < 20) or transition
                # Mean reversion at Bollinger Bands with RSI confirmation
                if close[i] <= bb_lower[i] and rsi_values[i] < 30 and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper[i] and rsi_values[i] > 70 and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long position: exit conditions
            if adx_val > 25:  # Trending: exit on trend reversal or overextension
                if close[i] < ema_20[i] * 0.98 or rsi_values[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit at mean or opposite extreme
                if close[i] >= sma_20[i] or rsi_values[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Short position: exit conditions
            if adx_val > 25:  # Trending: exit on trend reversal or overextension
                if close[i] > ema_20[i] * 1.02 or rsi_values[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit at mean or opposite extreme
                if close[i] <= sma_20[i] or rsi_values[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals