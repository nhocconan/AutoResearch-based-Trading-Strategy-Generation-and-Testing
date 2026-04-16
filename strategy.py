# This strategy uses a 4h Bollinger Squeeze with 1d RSI momentum and volume confirmation.
# In low volatility (Bollinger Band Width < 20th percentile), we look for RSI(14) extremes on the daily timeframe.
# Long when RSI < 30 (oversold) and price breaks above the Bollinger middle band with volume.
# Short when RSI > 70 (overbought) and price breaks below the Bollinger middle band with volume.
# Position size 0.25 to manage drawdown. Designed to capture mean reversion after volatility contractions
# in both bull and bear markets, avoiding whipsaw in high volatility periods.

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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for RSI) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 4h Bollinger Bands (20, 2) ===
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).mean()
    std_bb = pd.Series(close_4h).rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = sma_bb + (std_bb * bb_std)
    bb_lower = sma_bb - (std_bb * bb_std)
    bb_middle = sma_bb  # same as SMA
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (20) for squeeze detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True) * 100
    squeeze = bb_width_percentile < 20  # True when BBW in lowest 20%
    
    # === 1d RSI(14) ===
    rsi_period = 14
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === 4h volume ratio for confirmation ===
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    
    # Align all indicators to 4h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_4h, squeeze.values.astype(float))
    bb_middle_aligned = align_htf_to_ltf(prices, df_4h, bb_middle.values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        is_squeeze = squeeze_aligned[i] > 0.5  # True if in squeeze
        bb_mid = bb_middle_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        
        # === STOPLOSS LOGIC (fixed 4% stop) ===
        if position == 1:  # Long position
            if price < entry_price * 0.96:  # 4% stop loss
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price * 1.04:  # 4% stop loss
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price crosses below BB middle or RSI > 60
            if price < bb_mid or rsi_val > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above BB middle or RSI < 40
            if price > bb_mid or rsi_val < 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            if is_squeeze and vol_ratio > 1.5:  # In squeeze with volume confirmation
                # LONG: RSI oversold and price breaks above BB middle
                if rsi_val < 30 and price > bb_mid:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # SHORT: RSI overbought and price breaks below BB middle
                elif rsi_val > 70 and price < bb_mid:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_BollingerSqueeze_RSI_Momentum_Volume_v1"
timeframe = "4h"
leverage = 1.0