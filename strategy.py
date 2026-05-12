# 1D_TopBottom_Reversal_WeeklyTrend
# Strategy: Buy at weekly support levels (pivot points) when price shows reversal signals on daily timeframe.
# Sell/short at weekly resistance levels when price shows rejection.
# Uses weekly pivot points for structure, daily RSI for momentum, and volume confirmation.
# Designed to work in both bull and bear markets by trading mean reversion within the weekly trend.
# Target: 10-30 trades per year to minimize fee drag.

name = "1D_TopBottom_Reversal_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

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

    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C)/3
    # Support 1 = (2*Pivot) - High
    # Resistance 1 = (2*Pivot) - Low
    # Support 2 = Pivot - (High - Low)
    # Resistance 2 = Pivot + (High - Low)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    pivot = (wk_high + wk_low + wk_close) / 3.0
    support1 = (2 * pivot) - wk_high
    resistance1 = (2 * pivot) - wk_low
    support2 = pivot - (wk_high - wk_low)
    resistance2 = pivot + (wk_high - wk_low)
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    support1_aligned = align_htf_to_ltf(prices, df_1w, support1)
    resistance1_aligned = align_htf_to_ltf(prices, df_1w, resistance1)
    support2_aligned = align_htf_to_ltf(prices, df_1w, support2)
    resistance2_aligned = align_htf_to_ltf(prices, df_1w, resistance2)
    
    # Weekly trend: price above/below pivot
    wk_trend_up = wk_close > pivot
    wk_trend_down = wk_close < pivot
    wk_trend_up_aligned = align_htf_to_ltf(prices, df_1w, wk_trend_up)
    wk_trend_down_aligned = align_htf_to_ltf(prices, df_1w, wk_trend_down)
    
    # Daily RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(support1_aligned[i]) or 
            np.isnan(resistance1_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price near weekly support in uptrend with bullish momentum
            near_support = (low[i] <= support1_aligned[i] * 1.02) or (low[i] <= support2_aligned[i] * 1.02)
            bullish_momentum = rsi[i] > 30 and rsi[i] < 70  # Not oversold/overbought
            volume_ok = volume[i] > vol_avg_20[i] * 1.5
            weekly_uptrend = wk_trend_up_aligned[i]
            
            if near_support and bullish_momentum and volume_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly resistance in downtrend with bearish momentum
            elif (high[i] >= resistance1_aligned[i] * 0.98) or (high[i] >= resistance2_aligned[i] * 0.98):
                near_resistance = True
            else:
                near_resistance = False
                
            bearish_momentum = rsi[i] < 70 and rsi[i] > 30  # Not overbought/oversold
            volume_ok = volume[i] > vol_avg_20[i] * 1.5
            weekly_downtrend = wk_trend_down_aligned[i]
            
            if near_resistance and bearish_momentum and volume_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # EXIT LONG: Price reaches weekly resistance or momentum turns bearish
            at_resistance = (high[i] >= resistance1_aligned[i] * 0.98) or (high[i] >= resistance2_aligned[i] * 0.98)
            momentum_bearish = rsi[i] > 70  # Overbought
            if at_resistance or momentum_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # EXIT SHORT: Price reaches weekly support or momentum turns bullish
            at_support = (low[i] <= support1_aligned[i] * 1.02) or (low[i] <= support2_aligned[i] * 1.02)
            momentum_bullish = rsi[i] < 30  # Oversold
            if at_support or momentum_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals