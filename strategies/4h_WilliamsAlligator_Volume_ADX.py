# 2025-07-06: 4h Williams Alligator + Volume + ADX (4h ADX)
# Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# Long when Lips > Teeth > Jaw and price above Lips, with volume > 1.5x 20-bar volume MA and ADX > 20
# Short when Lips < Teeth < Jaw and price below Lips, with volume > 1.5x 20-bar volume MA and ADX > 20
# Exit when Alligator reverses (Lips crosses Teeth) or ADX drops below 15
# Fixed position size 0.25 to manage drawdown
# Designed for 4h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (all on 4h data)
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, smoothed 8 periods ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().shift(8)
    
    # Teeth: 8-period SMMA, smoothed 5 periods ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().shift(5)
    
    # Lips: 5-period SMMA, smoothed 3 periods ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().shift(3)
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # ADX (14-period) on 4h data
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx = adx.values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(adx[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Look for Alligator alignment with volume and ADX filter
            # Long: Lips > Teeth > Jaw (bullish alignment) and price above Lips
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and price > lips[i] and vol > 1.5 * vol_ma and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) and price below Lips
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and price < lips[i] and vol > 1.5 * vol_ma and adx[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Alligator turns bearish (Lips crosses below Teeth) or ADX weak
            if lips[i] < teeth[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Alligator turns bullish (Lips crosses above Teeth) or ADX weak
            if lips[i] > teeth[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Volume_ADX"
timeframe = "4h"
leverage = 1.0