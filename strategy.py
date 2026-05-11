#!/usr/bin/env python3
name = "1d_Keltner_Squeeze_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 2. Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 3. Daily ATR for Keltner Channels
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 4. Daily EMA20 for Keltner middle
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 5. Keltner Channels
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # 6. Bollinger Bands for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    
    # 7. Squeeze condition: BB inside Keltner
    squeeze = (bb_upper <= keltner_upper) & (bb_lower >= keltner_lower)
    
    # 8. Momentum: 12-period ROC
    roc = np.zeros_like(close)
    roc[12:] = (close[12:] - close[:-12]) / close[:-12] * 100
    
    # 9. Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 10. Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(roc[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        in_squeeze = squeeze[i]
        momentum_up = roc[i] > 0
        momentum_down = roc[i] < 0
        price_above_keltner_upper = close[i] > keltner_upper[i]
        price_below_keltner_lower = close[i] < keltner_lower[i]
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: Squeeze breakout up + momentum up + weekly uptrend + volume
            if in_squeeze and momentum_up and price_above_keltner_upper and weekly_uptrend and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Squeeze breakout down + momentum down + weekly downtrend + volume
            elif in_squeeze and momentum_down and price_below_keltner_lower and weekly_downtrend and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price closes below Keltner lower OR momentum turns negative
                if close[i] < keltner_lower[i] or roc[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price closes above Keltner upper OR momentum turns positive
                if close[i] > keltner_upper[i] or roc[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals