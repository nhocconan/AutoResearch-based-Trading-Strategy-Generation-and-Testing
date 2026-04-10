#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1h
# - Trend filter: 4h EMA(21) slope (bullish if rising, bearish if falling)
# - Session filter: Trade only 08:00-20:00 UTC to avoid low-liquidity hours
# - Position size: 0.20 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 1h
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Camarilla pivots act as support/resistance; EMA filter ensures trend alignment

name = "1h_4h_camarilla_ema_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(21) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-compute 4h EMA(21) slope (trend direction)
    ema_slope = np.zeros_like(ema_4h_aligned)
    ema_slope[1:] = ema_4h_aligned[1:] - ema_4h_aligned[:-1]
    ema_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_slope)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_slope_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Camarilla mean reversion OR stoploss hit
            if close_1h[i] < camarilla_l3[i] or close_1h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Camarilla mean reversion OR stoploss hit
            if close_1h[i] > camarilla_h3[i] or close_1h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Calculate Camarilla levels for current 1h bar
            # Based on previous bar's high, low, close
            if i >= 1:
                high_prev = high_1h[i-1]
                low_prev = low_1h[i-1]
                close_prev = close_1h[i-1]
                range_prev = high_prev - low_prev
                
                camarilla_h3 = close_prev + range_prev * 1.1 / 4
                camarilla_l3 = close_prev - range_prev * 1.1 / 4
                
                # Look for Camarilla breakouts with trend and session filters
                bullish_trend = ema_slope_aligned[i] > 0  # Rising EMA = bullish
                bearish_trend = ema_slope_aligned[i] < 0  # Falling EMA = bearish
                
                # Long: price breaks above Camarilla H3 with bullish trend
                if close_1h[i] > camarilla_h3 and bullish_trend:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L3 with bearish trend
                elif close_1h[i] < camarilla_l3 and bearish_trend:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals