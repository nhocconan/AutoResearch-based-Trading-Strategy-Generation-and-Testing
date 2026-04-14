#1h_EMA21_ATR_Breakout_Volume_Session
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA trend filter, ATR breakout for entry, volume confirmation, and session filter (08-20 UTC).
# Long when price breaks above EMA21 + 0.5*ATR(14) with 4h EMA alignment (price > EMA) and volume > 1.5x average.
# Short when price breaks below EMA21 - 0.5*ATR(14) with 4h EMA alignment (price < EMA) and volume > 1.5x average.
# Exit when price returns to EMA21.
# Uses 4h EMA for trend filter, ATR for volatility-based breakout, volume for confirmation.
# Session filter reduces noise trades outside active hours.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA21 and ATR(14) on 1h data
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(21, 14, 20)  # Need EMA21, ATR14, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema21[i]) or 
            np.isnan(atr14[i]) or
            np.isnan(ema21_4h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 4h EMA21
        price_above_ema = close[i] > ema21_4h_aligned[i]
        price_below_ema = close[i] < ema21_4h_aligned[i]
        
        if position == 0:
            # Look for EMA21 breakouts with ATR buffer
            # Long: price breaks above EMA21 + 0.5*ATR AND price above 4h EMA
            if (close[i] > ema21[i] + 0.5 * atr14[i] and 
                price_above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below EMA21 - 0.5*ATR AND price below 4h EMA
            elif (close[i] < ema21[i] - 0.5 * atr14[i] and 
                  price_below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to EMA21
            if close[i] <= ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to EMA21
            if close[i] >= ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_EMA21_ATR_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0