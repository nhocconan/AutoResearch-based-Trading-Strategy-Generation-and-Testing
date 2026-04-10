#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w EMA trend filter + volume confirmation
# - Williams Alligator: Jaw (EMA13, shift8), Teeth (EMA8, shift5), Lips (EMA5, shift3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1w close > 1w EMA(34) AND 12h volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1w close < 1w EMA(34) AND 12h volume > 1.5x 20-period average
# - Exit when Alligator lines re-cross (Lips crosses Teeth)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Alligator identifies trend emergence; weekly EMA filter ensures higher timeframe trend alignment
# - Volume confirmation reduces false breakouts
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1w_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1w) < 50 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Williams Alligator
    # Jaw: EMA(13, shift=8)
    jaw = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 13:
        jaw[12] = np.mean(close[:13])  # SMA seed
        for i in range(13, len(close)):
            jaw[i] = (close[i] * 2 + jaw[i-1] * 12) / 14  # EMA(13)
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    
    # Teeth: EMA(8, shift=5)
    teeth = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 8:
        teeth[7] = np.mean(close[:8])  # SMA seed
        for i in range(8, len(close)):
            teeth[i] = (close[i] * 2 + teeth[i-1] * 7) / 10  # EMA(8)
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    
    # Lips: EMA(5, shift=3)
    lips = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 5:
        lips[4] = np.mean(close[:5])  # SMA seed
        for i in range(5, len(close)):
            lips[i] = (close[i] * 2 + lips[i-1] * 4) / 6  # EMA(5)
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Pre-compute 12h ATR (14-period) for dynamic stop
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 12h volume average (20-period)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_12h = rolling_mean(volume, 20)
    
    # Pre-compute 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_34_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])  # SMA seed
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 32) / 34  # EMA(34)
    
    # Align HTF indicators to 12h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition
            vol_spike = not np.isnan(vol_ma_12h_aligned[i]) and volume[i] > 1.5 * vol_ma_12h_aligned[i]
            
            # Long conditions: Alligator bullish alignment AND volume spike AND 1w uptrend
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike and 
                close[i] > ema_34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Alligator bearish alignment AND volume spike AND 1w downtrend
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike and 
                  close[i] < ema_34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator lines re-cross (Lips crosses Teeth)
            exit_long = (position == 1 and lips[i] <= teeth[i])
            exit_short = (position == -1 and lips[i] >= teeth[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= high[i] - 2.5 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 2.5 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

if __name__ == "__main__":
    # Quick sanity check
    print("Strategy: 12h Williams Alligator + 1w EMA + Volume")
    print("Timeframe:", timeframe)
    print("Leverage:", leverage)
    print("Discrete position sizing: ±0.25")
    print("Entry conditions: Alligator alignment + volume spike + weekly EMA filter")
    print("Exit conditions: Alligator re-cross or ATR stop")