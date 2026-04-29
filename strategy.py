#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation
# Long when Alligator jaws (13) > teeth (8) > lips (5) in bullish 1w trend (price > EMA34) with volume spike
# Short when jaws < teeth < lips in bearish 1w trend (price < EMA34) with volume spike
# Uses weekly EMA34 to filter for strong trends only, avoiding whipsaws in ranging conditions
# Volume confirmation ensures moves have institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_WilliamsAlligator_1wEMA34_VolumeSpike_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 12h timeframe (completed 1w bar only)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator on 12h: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    # SMMA = smoothed moving average (Wilder's smoothing)
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # shift forward
    jaw[:8] = np.nan
    
    # Teeth (red): 8-period SMMA, shifted 5 bars forward
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # shift forward
    teeth[:5] = np.nan
    
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # shift forward
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: price relative to weekly EMA34
        is_bullish_trend = curr_close > curr_ema_34_1w
        is_bearish_trend = curr_close < curr_ema_34_1w
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and clear Alligator alignment
            if curr_volume_confirm:
                # Bullish Alligator: jaw > teeth > lips in bullish weekly trend
                if (curr_jaw > curr_teeth > curr_lips) and is_bullish_trend:
                    signals[i] = 0.25
                    position = 1
                # Bearish Alligator: jaw < teeth < lips in bearish weekly trend
                elif (curr_jaw < curr_teeth < curr_lips) and is_bearish_trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Alligator loses bullish alignment OR weekly trend turns bearish
            if not (curr_jaw > curr_teeth > curr_lips) or not is_bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Alligator loses bearish alignment OR weekly trend turns bullish
            if not (curr_jaw < curr_teeth < curr_lips) or not is_bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals