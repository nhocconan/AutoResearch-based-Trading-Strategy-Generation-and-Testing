#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA(50) trend filter and volume confirmation
# - Uses 1h Camarilla pivot levels (H3/L3) for breakout entries
# - Trend filter: 4h EMA(50) to ensure trades align with higher timeframe trend
# - Volume confirmation: volume > 1.5x 20-period average to filter weak breakouts
# - Session filter: 08-20 UTC to avoid low-liquidity periods
# - Position size: 0.20 (20% of capital) - discrete level to minimize fee churn
# - Target: ~20-30 trades/year (80-120 total over 4 years) to stay under fee drag threshold
# - Camarilla pivots work well in ranging markets, EMA filter avoids counter-trend trades

name = "1h_camarilla_ema4h_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Camarilla pivot levels (based on previous day's OHLC)
    # We'll use rolling window of 24 periods (1 day) to calculate pivots
    lookback = 24
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last()
    
    # Camarilla levels
    camarilla_h3 = roll_close + (roll_high - roll_low) * 1.1 / 4
    camarilla_l3 = roll_close - (roll_high - roll_low) * 1.1 / 4
    camarilla_h4 = roll_close + (roll_high - roll_low) * 1.1 / 2
    camarilla_l4 = roll_close - (roll_high - roll_low) * 1.1 / 2
    
    # 1h volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i]) or
            np.isnan(volume_spike[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retraces below Camarilla H3
            if close[i] < camarilla_h3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retraces above Camarilla L3
            if close[i] > camarilla_l3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and 4h EMA trend filter
            # Long: price breaks above Camarilla H4 AND price > 4h EMA50 AND volume spike AND in session
            if (close[i] >= camarilla_h4[i] and 
                close[i] > ema_50_4h_aligned[i] and
                volume_spike[i] and
                in_session[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below Camarilla L4 AND price < 4h EMA50 AND volume spike AND in session
            elif (close[i] <= camarilla_l4[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  volume_spike[i] and
                  in_session[i]):
                position = -1
                signals[i] = -0.20
    
    return signals