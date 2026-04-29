#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based stoploss
# Long when price breaks above Donchian upper band AND 1d EMA34 uptrend AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower band AND 1d EMA34 downtrend AND volume > 2.0x 20-period average
# Exit when price crosses Donchian midline (10-period average of high/low) OR ATR stoploss hit
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing strong trends

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
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2  # Midline for exit
    
    # ATR(14) for stoploss and position sizing reference
    period_atr = 14
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).max().values - pd.Series(close).shift(1).values)
    tr3 = abs(pd.Series(low).rolling(window=2).min().values - pd.Series(close).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period_atr, min_periods=period_atr).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(period_dc, 34, 20, period_atr)  # warmup for Donchian, EMA34, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_dc_middle = donchian_middle[i]
        curr_ema34 = ema34_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price > Donchian upper AND 1d EMA34 uptrend
                if curr_close > curr_dc_upper and curr_close > curr_ema34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price < Donchian lower AND 1d EMA34 downtrend
                elif curr_close < curr_dc_lower and curr_close < curr_ema34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price < Donchian middle OR ATR stoploss hit (2.0 * ATR below entry)
            stop_price = entry_price - 2.0 * curr_atr
            if curr_close < curr_dc_middle or curr_low <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price > Donchian middle OR ATR stoploss hit (2.0 * ATR above entry)
            stop_price = entry_price + 2.0 * curr_atr
            if curr_close > curr_dc_middle or curr_high >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals