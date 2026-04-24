#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H4/L4 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter and volume MA confirmation.
- Entry: Long when price breaks above Camarilla H4 AND price > 4h EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L4 AND price < 4h EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 4h EMA34).
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Uses session filter (08-20 UTC) to reduce noise trades.
- Camarilla levels provide intraday support/resistance; 4h EMA34 filters counter-trend breakouts.
- Works in bull markets via breakout continuation and bear markets via trend-filtered mean reversion at Camarilla levels.
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
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla (using previous day's OHLC)
    typical_price = (high + low + close) / 3.0
    range_val = high - low
    
    # Calculate Camarilla levels (H4, L4) based on previous day's data
    # H4 = close + 1.1 * (high - low) 
    # L4 = close - 1.1 * (high - low)
    camarilla_h4 = close + 1.1 * range_val
    camarilla_l4 = close - 1.1 * range_val
    
    # Shift to use previous day's levels (lookback by 1 bar)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    
    # Get 4h data for EMA34 and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate EMA(34) on 4h
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 4h EMA34 (trend filter)
        if position == 1:
            if curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > camarilla_h4[i]
        bearish_breakout = curr_close < camarilla_l4[i]
        
        # Trend filter from 4h EMA34
        price_above_ema = curr_close > ema_34_aligned[i]
        price_below_ema = curr_close < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: breakout above H4 AND price above 4h EMA34
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.20
                    position = 1
                # Short: breakout below L4 AND price below 4h EMA34
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H4L4_Breakout_4hEMA34_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0