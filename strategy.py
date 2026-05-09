#!/usr/bin/env python3
# 6H_1W_1D_OrderBlock_Reversal_Trend
# Hypothesis: On 6h timeframe, identify institutional order blocks from weekly candles and trade reversals
# when price returns to these blocks with 1d trend confirmation. In bull markets, buy at demand zones;
# in bear markets, sell at supply zones. Uses weekly structure for institutional levels and daily
# trend filter to avoid counter-trend trades. Targets 15-35 trades/year per symbol.

name = "6H_1W_1D_OrderBlock_Reversal_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for order blocks (institutional supply/demand zones)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly order blocks: bullish (demand) and bearish (supply) zones
    # Bullish OB: prior bearish candle followed by bullish candle - use low of bearish candle as demand zone
    # Bearish OB: prior bullish candle followed by bearish candle - use high of bullish candle as supply zone
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Identify bullish order blocks (demand zones)
    weekly_bearish = weekly_close < weekly_open  # prior candle bearish
    weekly_bullish = weekly_close > weekly_open  # current candle bullish
    bullish_ob = weekly_bearish & weekly_bullish  # bearish then bullish = demand zone
    ob_demand = np.where(bullish_ob, weekly_low, np.nan)  # low of bearish candle
    
    # Identify bearish order blocks (supply zones)
    weekly_bullish_prior = weekly_close > weekly_open  # prior candle bullish
    weekly_bearish_curr = weekly_close < weekly_open   # current candle bearish
    bearish_ob = weekly_bullish_prior & weekly_bearish_curr  # bullish then bearish = supply zone
    ob_supply = np.where(bearish_ob, weekly_high, np.nan)  # high of bullish candle
    
    # Forward fill to create persistent zones until next OB of same type
    ob_demand_series = pd.Series(ob_demand)
    ob_demand_ffilled = ob_demand_series.ffill().values
    
    ob_supply_series = pd.Series(ob_supply)
    ob_supply_ffilled = ob_supply_series.ffill().values
    
    # Daily trend filter: EMA(34) on close
    daily_close = df_1d['close'].values
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = daily_close > ema_34
    
    # Align weekly order blocks and daily trend to 6h
    ob_demand_aligned = align_htf_to_ltf(prices, df_1w, ob_demand_ffilled)
    ob_supply_aligned = align_htf_to_ltf(prices, df_1w, ob_supply_ffilled)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ob_demand_aligned[i]) or np.isnan(ob_supply_aligned[i]) or 
            np.isnan(trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price returns to weekly demand zone (bullish OB) with daily uptrend
            if (close[i] <= ob_demand_aligned[i] * 1.005 and  # within 0.5% of demand zone
                close[i] >= ob_demand_aligned[i] * 0.995 and
                trend_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price returns to weekly supply zone (bearish OB) with daily downtrend
            elif (close[i] <= ob_supply_aligned[i] * 1.005 and  # within 0.5% of supply zone
                  close[i] >= ob_supply_aligned[i] * 0.995 and
                  not trend_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches supply zone or trend turns down
            if (close[i] >= ob_supply_aligned[i] * 0.995 or  # reached supply zone
                not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches demand zone or trend turns up
            if (close[i] <= ob_demand_aligned[i] * 1.005 or  # reached demand zone
                trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals