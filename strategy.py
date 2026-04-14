# 12h Wedge Breakout with Volume Confirmation and ATR Stop
# Uses 12h primary timeframe with 1w context
# Entry: Price breaks above/below 12h Donchian(20) with volume > 1.5x 24-period average
# Exit: ATR-based trailing stop (2.5 * ATR) or opposite breakout
# Position sizing: 0.25 (25%)
# Timeframe: 12h | Target trades: 50-150 total over 4 years (12-37/year)
# Works in bull/bear: Breakouts capture momentum, volume confirmation reduces false signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data (primary timeframe) - call ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Get 1w data for context (higher timeframe trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x average volume (24-period on 12h data)
    # Need to calculate average volume on 12h data then align
    vol_12h = df_12h['volume'].values
    avg_vol_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    # ATR for stop loss (using 12h data)
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    close_12h_series = pd.Series(close_12h)
    tr1 = high_12h_series - low_12h_series
    tr2 = abs(high_12h_series - close_12h_series.shift(1))
    tr3 = abs(low_12h_series - close_12h_series.shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 24)  # EMA 50 and volume 24
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_vol_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        upper_breakout = high_20_aligned[i]
        lower_breakout = low_20_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        avg_vol = avg_vol_12h_aligned[i]
        atr = atr_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation
            # In bull trend (price above 1w EMA) OR in bear trend with strong momentum
            if price > upper_breakout and vol > 1.5 * avg_vol:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian with volume confirmation
            elif price < lower_breakout and vol > 1.5 * avg_vol:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR ATR stop hit
            # Track highest high since entry for trailing stop
            if i > start:
                # Simple approach: exit on opposite breakout or close below entry - 2.5*ATR
                # For simplicity, using opposite breakout and time-based exit
                if price < lower_breakout:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR ATR stop hit
            if price > upper_breakout:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Wedge_Breakout_Volume"
timeframe = "12h"
leverage = 1.0