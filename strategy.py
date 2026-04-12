# 4h_1d_camarilla_breakout_volume_v2
# Hypothesis: Use 1-day Camarilla pivot levels with volume confirmation and price above/below 200 EMA for trend filter. 
# The strategy breaks out from Camarilla levels (H3/L3) with volume confirmation in the direction of the 200 EMA trend.
# Works in bull markets (buy breakouts above H3 in uptrend) and bear markets (sell breakdowns below L3 in downtrend).
# Volume filter reduces false breakouts. Position size scaled inversely to volatility (ATR-based) to manage risk.
# Target: 20-40 trades/year per symbol with discrete position sizing to minimize fee churn.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR (14-period) for volatility-based position sizing
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period ATR mean for volatility ratio
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma_20
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volatility-based position sizing (inverse volatility)
    # Higher volatility = smaller position, capped between 0.10 and 0.30
    vol_scaling = np.clip(1.0 / (atr_ratio_aligned + 0.001), 0.5, 2.0)
    base_size = 0.25
    position_size = base_size * vol_scaling
    position_size = np.clip(position_size, 0.10, 0.30)
    
    # Calculate Camarilla levels (H3, L3) using previous day's data
    camarilla_high = np.full(len(close_1d), np.nan)
    camarilla_low = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        camarilla_high[i] = C + ((H - L) * 1.1 / 2)  # H3 level
        camarilla_low[i] = C - ((H - L) * 1.1 / 2)   # L3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Calculate 200 EMA on 4h for trend filter
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    price_above_ema = close > ema_200
    price_below_ema = close < ema_200
    
    # Volume filter: current volume > 20-period average (on 4h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(position_size[i]) or
            np.isnan(price_above_ema[i]) or np.isnan(price_below_ema[i])):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Breakout conditions with volume and trend confirmation
        breakout_up = close[i] > camarilla_high_aligned[i] and price_above_ema[i]
        breakout_down = close[i] < camarilla_low_aligned[i] and price_below_ema[i]
        vol_ok = volume_ok[i]
        
        # Entry signals
        long_signal = breakout_up and vol_ok
        short_signal = breakout_down and vol_ok
        
        # Exit when price returns to the 1-day close (punishment level)
        # Use previous day's close as exit level
        prev_close = np.full(len(close_1d), np.nan)
        for j in range(1, len(close_1d)):
            prev_close[j] = close_1d[j-1]
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
        
        exit_long = close[i] < prev_close_aligned[i]
        exit_short = close[i] > prev_close_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size[i]
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size[i]
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position with dynamic sizing
            if position == 1:
                signals[i] = position_size[i]
            elif position == -1:
                signals[i] = -position_size[i]
            else:
                signals[i] = 0.0
    
    return signals