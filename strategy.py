#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with 12h EMA trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA, helping identify institutional buying/selling pressure.
# In bull markets: buy when Bull Power > 0 and rising with volume spike and 12h uptrend.
# In bear markets: sell when Bear Power < 0 and falling with volume spike and 12h downtrend.
# Uses discrete position sizing (0.25) to minimize fee drag and ensure low trade frequency (<25/year).
# Designed to work in both bull and bear markets by adapting to trend direction via 12h EMA filter.

name = "6h_ElderRay_BullBearPower_12hTrend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h_s = pd.Series(df_12h['close'].values)
    ema_34_12h = close_12h_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for 12h EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_12h = ema_34_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: Bull Power > 0 and rising with 12h uptrend
                if curr_bull_power > 0 and curr_bull_power > bull_power[i-1] and curr_close > curr_ema_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Bear Power < 0 and falling with 12h downtrend
                elif curr_bear_power < 0 and curr_bear_power < bear_power[i-1] and curr_close < curr_ema_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Elder Ray turns negative
            # Calculate ATR(14) dynamically for stoploss
            tr1 = high[max(0, i-14):i+1] - low[max(0, i-14):i+1]
            tr2 = np.abs(high[max(0, i-14):i+1] - np.concatenate([[close[0]], close[max(0, i-14):i]]))
            tr3 = np.abs(low[max(0, i-14):i+1] - np.concatenate([[close[0]], close[max(0, i-14):i]]))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            
            if curr_close < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif curr_bull_power <= 0:  # Bull Power turned negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Elder Ray turns positive
            tr1 = high[max(0, i-14):i+1] - low[max(0, i-14):i+1]
            tr2 = np.abs(high[max(0, i-14):i+1] - np.concatenate([[close[0]], close[max(0, i-14):i]]))
            tr3 = np.abs(low[max(0, i-14):i+1] - np.concatenate([[close[0]], close[max(0, i-14):i]]))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            
            if curr_close > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif curr_bear_power >= 0:  # Bear Power turned positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals