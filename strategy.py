#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + ATR volatility filter
# - Long when price breaks above Camarilla H3 level with volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level with volume > 1.5x 20-period average
# - ATR filter: only trade when ATR(10) > 0.3 * ATR(40) to avoid low volatility chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts above H3/H4) and bear (breakdowns below L3/L4) markets
# - 1d HTF provides reliable pivot levels and volume confirmation

name = "4h_1d_camarilla_volume_atr_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_h3 = close_1d + 1.125 * range_1d
    camarilla_l3 = close_1d - 1.125 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d ATR for volatility filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_40_1d = pd.Series(tr_1d).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    atr_40_aligned = align_htf_to_ltf(prices, df_1d, atr_40_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_10_aligned[i]) or np.isnan(atr_40_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_h3_aligned[i-1]  # Close above previous period's H3
        breakout_short = price_close < camarilla_l3_aligned[i-1]  # Close below previous period's L3
        
        # Volume confirmation: current volume > 1.5x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # ATR filter: trade only when short-term ATR > 0.3 * long-term ATR (avoid low volatility)
        atr_filter = atr_10_aligned[i] > 0.3 * atr_40_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + ATR filter
        if breakout_long and vol_confirm and atr_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + ATR filter
        if breakout_short and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level break or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR volatility collapses
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (not atr_filter)
        elif position == -1:
            # Exit short if price breaks above H3 OR volatility collapses
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (not atr_filter)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals