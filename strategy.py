#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter to avoid counter-trend trades.
- Entry: Long when price breaks above Donchian upper band (20-period high) AND 1w EMA50 > 1w EMA50(previous) (uptrend);
         Short when price breaks below Donchian lower band (20-period low) AND 1w EMA50 < 1w EMA50(previous) (downtrend).
- Exit: Close-based reversal (opposite signal) or ATR stoploss (signal=0 when price moves against position by 2.5 * ATR(14)).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian channels provide clear trend-following structure; 1w EMA50 ensures we only trade with the dominant weekly trend;
  ATR stoploss manages risk during volatile periods. Works in bull markets (buy breakouts in uptrend) and bear markets
  (sell breakdowns in downtrend) with trend filter to avoid whipsaws.
- Estimated trades: ~50 total over 4 years (~12/year) based on Donchian(20) breakout frequency with weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_1w - np.roll(ema_50_1w, 1)
    ema_50_slope[0] = 0
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to primary 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Update entry price when position is opened
        if position != 0 and entry_price == 0.0:
            entry_price = curr_close
        
        # Exit conditions
        if position != 0:
            # ATR-based stoploss
            if position == 1 and curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1 and curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            
            # Trend filter exit: opposite weekly EMA50 slope
            if position == 1 and ema_50_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            elif position == -1 and ema_50_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions with weekly trend filter
        bullish_breakout = curr_high > donchian_high[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_low[i]    # Break below lower band
        
        # Weekly trend filter
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        if position == 0:
            # Check for entry signals
            if bullish_breakout and uptrend:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif bearish_breakout and downtrend:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_EMA50_Trend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0