# Hypothesis: 4h Bollinger Band reversal with volume confirmation and daily trend filter
# In bull markets, buy BB lower band bounce; in bear markets, sell BB upper band rejection
# Volume confirms momentum, daily trend prevents counter-trend trades
# Target: 20-40 trades/year to minimize fee drag
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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 4h Bollinger Bands (20, 2.0) ===
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    
    # === 1d EMA (50) for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h volume ratio (20) ===
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_4h / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio)
    
    signals = np.zeros(n)
    
    # Warmup: enough for BB and EMA
    warmup = 60
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price touches or crosses upper BB or 2% trailing stop
            if price >= upper or price < entry_price * 0.98:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price touches or crosses lower BB or 2% trailing stop
            if price <= lower or price > entry_price * 1.02:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price at/below lower BB, below daily EMA (bearish bias), strong volume
            if price <= lower and price < ema_trend and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: price at/above upper BB, above daily EMA (bullish bias), strong volume
            elif price >= upper and price > ema_trend and vol_ratio_val > 1.5:
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

name = "4h_BB_Reversal_Trend_Volume"
timeframe = "4h"
leverage = 1.0