#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > 1d EMA34 AND volume > 1.5x 20-period average volume
# Short when Bear Power < 0 (low < EMA13) AND Bull Power < 0 (close < EMA13) AND price < 1d EMA34 AND volume > 1.5x 20-period average volume
# No trailing stop - rely on signal reversal for exit to minimize whipsaw
# Designed for moderate trade frequency (target: 80-150 total trades over 4 years) to balance signal quality and cost
# Elder Ray captures market strength/weakness relative to EMA13
# EMA34 filter ensures alignment with medium-term trend to avoid counter-trend trades
# Volume confirmation adds conviction to signals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h EMA13 (Elder Ray base) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # === 6h Bull Power (close - EMA13) ===
    bull_power = close_6h - ema_13_6h
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    
    # === 6h Bear Power (low - EMA13) ===
    bear_power = low_6h - ema_13_6h
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # === 6h Volume Spike Confirmation (20-period average) ===
    vol_6h = df_6h['volume'].values
    vol_ma_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_13_6h_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for spike
        
        # === ENTRY LOGIC ===
        # Long when: Bull Power > 0 AND Bear Power < 0 (strong bullish) AND price > 1d EMA34 AND volume spike
        if bull_val > 0 and bear_val < 0 and price > ema_34_val and vol_confirm:
            signals[i] = 0.25
        # Short when: Bear Power < 0 AND Bull Power < 0 (strong bearish) AND price < 1d EMA34 AND volume spike
        elif bear_val < 0 and bull_val < 0 and price < ema_34_val and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0