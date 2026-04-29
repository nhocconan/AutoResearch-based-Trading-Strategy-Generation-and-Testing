#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation, and ATR stoploss
# Long when price breaks above Donchian upper band AND price > 1d EMA34 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND price < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when price touches Donchian middle band (10-period average of high/low) or ATR-based stoploss
# Works in both bull/bear by following 1d trend. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian Channel (20) on 4h data
    period_dc = 20
    highest_high = pd.Series(high).rolling(window=period_dc, min_periods=period_dc).max().values
    lowest_low = pd.Series(low).rolling(window=period_dc, min_periods=period_dc).min().values
    upper_band = highest_high
    lower_band = lowest_low
    middle_band = (highest_high + lowest_low) / 2  # Donchian middle (10-period avg of high/low)
    
    # ATR(14) for stoploss
    period_atr = 14
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).max().values - pd.Series(close).shift(1).values)
    tr3 = abs(pd.Series(low).rolling(window=2).min().values - pd.Series(close).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period_atr, min_periods=period_atr).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(period_dc, 34, period_atr, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_middle = middle_band[i]
        curr_ema34 = ema34_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper band AND price > 1d EMA34
                if curr_close > curr_upper and curr_close > curr_ema34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below lower band AND price < 1d EMA34
                elif curr_close < curr_lower and curr_close < curr_ema34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price touches middle band OR ATR stoploss hit OR price < 1d EMA34
            if (curr_low <= curr_middle) or (curr_close <= entry_price - 2.0 * curr_atr) or (curr_close < curr_ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price touches middle band OR ATR stoploss hit OR price > 1d EMA34
            if (curr_high >= curr_middle) or (curr_close >= entry_price + 2.0 * curr_atr) or (curr_close > curr_ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals