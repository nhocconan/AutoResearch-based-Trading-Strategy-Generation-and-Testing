#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe for structure) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4x ATR for volatility filter (14-period)
    high_low_4h = high_4h - low_4h
    high_close_4h = np.abs(high_4h - np.roll(close_4h, 1))
    low_close_4h = np.abs(low_4h - np.roll(close_4h, 1))
    high_close_4h[0] = np.inf
    low_close_4h[0] = np.inf
    tr_4h = np.maximum(high_low_4h, np.maximum(high_close_4h, low_close_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4h Bollinger Bands (20-period, 2 std)
    close_4h_series = pd.Series(close_4h)
    sma_20_4h = close_4h_series.rolling(window=20, min_periods=20).mean().values
    std_20_4h = close_4h_series.rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma_20_4h + (2 * std_20_4h)
    lower_bb_4h = sma_20_4h - (2 * std_20_4h)
    upper_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h)
    lower_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h)
    
    # === 1d data (higher timeframe for regime filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Bollinger Band Width (20-period, 2 std) for regime detection
    close_1d_series = pd.Series(close_1d)
    sma_20_1d = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std_20_1d = close_1d_series.rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d  # Normalized width
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # === 4h indicators for entry timing ===
    # RSI(14) on 4h close
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume spike detection (4h)
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # Session filter: 08-20 UTC (aligned with institutional activity)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(upper_bb_4h_aligned[i]) or 
            np.isnan(lower_bb_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(bb_width_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        atr_4h_val = atr_4h_aligned[i]
        upper_bb_4h_val = upper_bb_4h_aligned[i]
        lower_bb_4h_val = lower_bb_4h_aligned[i]
        rsi_4h_val = rsi_4h_aligned[i]
        vol_ratio_4h_val = vol_ratio_4h_aligned[i]
        bb_width_1d_val = bb_width_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below lower Bollinger Band OR RSI becomes overbought
            if (price < lower_bb_4h_val) or (rsi_4h_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above upper Bollinger Band OR RSI becomes oversold
            if (price > upper_bb_4h_val) or (rsi_4h_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price touches lower Bollinger Band (mean reversion) 
                # AND RSI is oversold (<30) AND volume spike AND low volatility regime (BB width < 30th percentile)
                if (price <= lower_bb_4h_val) and (rsi_4h_val < 30) and \
                   (vol_ratio_4h_val > 1.5) and (bb_width_1d_val < np.percentile(bb_width_1d_aligned[:i+1], 30)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price touches upper Bollinger Band (mean reversion) 
                # AND RSI is overbought (>70) AND volume spike AND low volatility regime
                elif (price >= upper_bb_4h_val) and (rsi_4h_val > 70) and \
                     (vol_ratio_4h_val > 1.5) and (bb_width_1d_val < np.percentile(bb_width_1d_aligned[:i+1], 30)):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Bounce_RSI_Volume"
timeframe = "4h"
leverage = 1.0