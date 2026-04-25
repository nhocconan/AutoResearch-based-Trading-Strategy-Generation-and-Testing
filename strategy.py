#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla pivot levels (R1/S1) act as intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation and 1d EMA34 trend filter
capture momentum moves. Choppiness index filter avoids whipsaws in ranging markets.
Designed for 4h timeframe to balance trade frequency and signal quality.
Target: 20-50 trades/year (75-200 total over 4 years) to minimize fee drag.
Works in bull/bear via trend filter and volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Choppiness Index (14) for regime filter
    if len(close) >= 14:
        hl_range = pd.Series(high) - pd.Series(low)
        atr_14 = pd.Series(atr)  # ATR already calculated
        sum_atr = atr_14.rolling(window=14, min_periods=14).sum()
        max_hh = pd.Series(high).rolling(window=14, min_periods=14).max()
        min_ll = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(sum_atr / (max_hh - min_ll)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_aligned[i]
        atr_val = atr[i]
        chop_val = chop_values[i]
        
        # Calculate Camarilla levels from previous day
        if i >= 24:  # Need at least 24 4h bars (1 day) for previous day's OHLC
            # Get index of 24 bars ago (previous day close)
            prev_day_idx = i - 24
            if prev_day_idx >= 0:
                # Previous day's OHLC (24 bars ago close is yesterday's close)
                # We need the day before yesterday's close for today's Camarilla
                # Actually, for bar i, we want Camarilla from day before current bar's day
                # Simpler: use 24*2 = 48 bars ago for 2 days ago
                if i >= 48:
                    day_before_yesterday_idx = i - 48
                    if day_before_yesterday_idx >= 0:
                        # Get high, low, close from 2 days ago (24*2 bars back)
                        # But we need the previous day's complete OHLC
                        # Instead, use rolling window of 24 bars to get daily OHLC
                        lookback_start = max(0, i - 48)
                        lookback_end = i - 24
                        if lookback_end > lookback_start:
                            day_high = np.max(high[lookback_start:lookback_end])
                            day_low = np.min(low[lookback_start:lookback_end])
                            day_close = close[lookback_end - 1] if lookback_end > 0 else close[0]
                            
                            if day_high > day_low and day_close > 0:
                                range_val = day_high - day_low
                                camarilla_r1 = day_close + (range_val * 1.1 / 12)
                                camarilla_s1 = day_close - (range_val * 1.1 / 12)
                            else:
                                camarilla_r1 = curr_close
                                camarilla_s1 = curr_close
                        else:
                            camarilla_r1 = curr_close
                            camarilla_s1 = curr_close
                    else:
                        camarilla_r1 = curr_close
                        camarilla_s1 = curr_close
                else:
                    camarilla_r1 = curr_close
                    camarilla_s1 = curr_close
            else:
                camarilla_r1 = curr_close
                camarilla_s1 = curr_close
        else:
            camarilla_r1 = curr_close
            camarilla_s1 = curr_close
        
        # Volume spike: current volume > 1.8 * 30-period average
        if i >= 30:
            vol_ma_30 = np.mean(volume[i-29:i+1])
        else:
            vol_ma_30 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.8 * vol_ma_30
        
        # Trend filter
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        # Chop filter: avoid ranging markets (Chop > 61.8) and extreme trends (Chop < 38.2)
        # We want moderate chop: 38.2 <= Chop <= 61.8 for breakout trading
        chop_filter = (chop_val >= 38.2) and (chop_val <= 61.8)
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND volume spike AND uptrend AND chop filter
            long_condition = (curr_close > camarilla_r1) and volume_spike and uptrend and chop_filter
            # Short: price breaks below Camarilla S1 AND volume spike AND downtrend AND chop filter
            short_condition = (curr_close < camarilla_s1) and volume_spike and downtrend and chop_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or trend reversal or chop extreme
            if (curr_close <= entry_price - 2.0 * atr_val) or (not uptrend) or (chop_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or trend reversal or chop extreme
            if (curr_close >= entry_price + 2.0 * atr_val) or (not downtrend) or (chop_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0