#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 12h + volume spike + volatility regime filter
# - Long when price breaks above Camarilla H4 resistance with volume > 2.0x 20-period average (strong conviction)
# - Short when price breaks below Camarilla L4 support with volume > 2.0x 20-period average
# - Uses 12h data to calculate Camarilla levels (more stable than shorter timeframes)
# - Volatility filter: only trade when ATR(14) > ATR(50) to avoid low volatility chop
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits for 4h
# - Camarilla levels provide natural support/resistance based on previous day's price action
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets

name = "4h_12h_camarilla_volume_volatility_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formulas: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h4 = close_12h + (high_12h - low_12h) * 1.1 / 2
    camarilla_l4 = close_12h - (high_12h - low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # 12h volume SMA (20-period)
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    volume_sma_20_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Volatility filter: ATR(14) > ATR(50) to avoid low volatility chop
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_12h = pd.Series(tr_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_50_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h)
    
    # Pre-compute 4h price series for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_h4_aligned[i-1]  # Close above previous period's H4
        breakout_short = price_close < camarilla_l4_aligned[i-1]  # Close below previous period's L4
        
        # Volume confirmation: current volume > 2.0x 20-period average (using 12h aligned volume)
        vol_confirm = volume_current > 2.0 * volume_sma_20_aligned[i]
        
        # Volatility filter: trade only when short-term ATR > long-term ATR (avoid low volatility chop)
        vol_filter = atr_14_aligned[i] > atr_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H4 breakout + volume confirmation + volatility filter
        if breakout_long and vol_confirm and vol_filter:
            enter_long = True
        
        # Short: Camarilla L4 breakdown + volume confirmation + volatility filter
        if breakout_short and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L4 OR volatility filter fails
            exit_long = (price_close < camarilla_l4_aligned[i-1]) or (not vol_filter)
        elif position == -1:
            # Exit short if price breaks above H4 OR volatility filter fails
            exit_short = (price_close > camarilla_h4_aligned[i-1]) or (not vol_filter)
        
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