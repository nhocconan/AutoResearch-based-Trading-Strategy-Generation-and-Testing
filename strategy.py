#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Trend: 1d EMA34 (bullish if price > EMA34)
# Volume: current > 1.5x 20-period average
# Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 AND volume confirmed
# Short: Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND price < 1d EMA34 AND volume confirmed
# Exit: Opposite signal or ATR trailing stop (2.0 ATR)
# Target: 50-150 total trades over 4 years (12-37/year)
# Size: 0.25

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Elder Ray components: 13-period EMA of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for exit conditions (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        ema_trend = ema34_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Momentum conditions
        bullish_momentum = bp > 0 and br < 0  # Bull Power positive, Bear Power negative
        bearish_momentum = br < 0 and bp > 0  # Bear Power negative, Bull Power positive (same as above, but explicit)
        
        if position == 0:
            # Long: bullish momentum + uptrend + volume
            if bullish_momentum and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + downtrend + volume
            elif bearish_momentum and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish momentum or ATR trailing stop
            if bearish_momentum or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish momentum or ATR trailing stop
            if bullish_momentum or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals