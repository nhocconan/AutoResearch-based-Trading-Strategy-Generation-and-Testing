# 1d Candlestick Reversal with 1w Trend Filter and Volume Confirmation
# Takes long when a bullish engulfing pattern forms at support with weekly uptrend and volume spike
# Takes short when a bearish engulfing pattern forms at resistance with weekly downtrend and volume spike
# Exits when opposite engulfing pattern forms or trend weakens
# Designed to capture reversals in both bull and bear markets with confirmation
# Target: 30-100 trades over 4 years (7-25/year)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1w ATR for volatility-based support/resistance
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w_shifted)
    tr3 = np.abs(low_1w - close_1w_shifted)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate dynamic support/resistance levels
    # Support = weekly low - 0.5 * ATR (dynamic support)
    # Resistance = weekly high + 0.5 * ATR (dynamic resistance)
    support_level = np.minimum.reduce([low_1w, np.roll(low_1w, 1), np.roll(low_1w, 2)]) - 0.5 * atr_1w
    resistance_level = np.maximum.reduce([high_1w, np.roll(high_1w, 1), np.roll(high_1w, 2)]) + 0.5 * atr_1w
    
    # Align 1w indicators to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    support_level_aligned = align_htf_to_ltf(prices, df_1w, support_level)
    resistance_level_aligned = align_htf_to_ltf(prices, df_1w, resistance_level)
    
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(support_level_aligned[i]) or 
            np.isnan(resistance_level_aligned[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulfing = (close[i] > open_[i]) and (open_[i] < close[i-1]) and (close[i] > open_[i-1]) and (open_[i-1] > close[i-1])
        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulfing = (close[i] < open_[i]) and (open_[i] > close[i-1]) and (close[i] < open_[i-1]) and (open_[i-1] < close[i-1])
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1d[i] if vol_ma_1d[i] > 0 else 0
        
        if position == 0:
            # Long setup: bullish engulfing at support with weekly uptrend and volume spike
            if (bullish_engulfing and 
                price <= support_level_aligned[i] * 1.02 and  # Near support
                ema_1w_aligned[i] > np.roll(ema_1w_aligned, 5)[i] and  # Weekly uptrend
                vol_ratio > 1.5):                             # Volume spike
                position = 1
                signals[i] = position_size
            # Short setup: bearish engulfing at resistance with weekly downtrend and volume spike
            elif (bearish_engulfing and 
                  price >= resistance_level_aligned[i] * 0.98 and  # Near resistance
                  ema_1w_aligned[i] < np.roll(ema_1w_aligned, 5)[i] and  # Weekly downtrend
                  vol_ratio > 1.5):                             # Volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish engulfing forms or trend weakens
            if bearish_engulfing or ema_1w_aligned[i] < np.roll(ema_1w_aligned, 5)[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish engulfing forms or trend weakens
            if bullish_engulfing or ema_1w_aligned[i] > np.roll(ema_1w_aligned, 5)[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Candlestick_Reversal_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0