#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Uses 4h EMA(21) for trend direction (long when close > EMA, short when close < EMA)
# - Enters on 1h Camarilla H3/L3 breakout with volume confirmation (> 1.5x 20-period average)
# - Exits on opposite Camarilla breakout or time-based stop (max 12 bars)
# - Only trades during 08-20 UTC session to reduce noise
# - Discrete position sizing ±0.20 to limit drawdown and fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematical support/resistance levels that work in ranging markets
# - 4h EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation filters out weak breakouts
# - Session filter avoids low-liquidity periods

name = "1h_4h_camarilla_breakout_trend_v1"
timeframe = "1h"
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
    bars_in_trade = 0
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Pre-compute 4h EMA(21) for trend
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H3, L3, H4, L4
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not in_session[i]:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
                bars_in_trade = 0
            else:
                signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter from 4h EMA
        uptrend = close_price > ema_21_4h_aligned[i]
        downtrend = close_price < ema_21_4h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above H3 with volume confirmation and uptrend
        if close_price > h3 and vol_confirm and uptrend:
            enter_long = True
        
        # Short breakout: price closes below L3 with volume confirmation and downtrend
        if close_price < l3 and vol_confirm and downtrend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 (opposite Camarilla level) or max time reached
            exit_long = (close_price < l3) or (bars_in_trade >= 12)
        elif position == -1:
            # Exit short if price breaks above H3 (opposite Camarilla level) or max time reached
            exit_short = (close_price > h3) or (bars_in_trade >= 12)
        
        # Update bars in trade counter
        if position != 0:
            bars_in_trade += 1
        else:
            bars_in_trade = 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            bars_in_trade = 0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            bars_in_trade = 0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals