#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot reversal with 1w trend filter and volume confirmation
# - Primary signal: Price reverses from Camarilla H3/L3 levels on 1d
# - Trend filter: 1w EMA(21) slope confirms higher timeframe direction
# - Volume filter: 1d volume > 1.5x 20-period average volume (institutional participation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(14) on 1d
# - Target: 30-100 total trades over 4 years (7-25/year) per 1d strategy guidelines
# - Works in bull/bear: Camarilla levels adapt to volatility, trend filter avoids counter-trend

name = "1d_1w_camarilla_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_slope = np.diff(ema_21, prepend=ema_21[0])  # positive = uptrend
    ema_21_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_21_slope)
    
    # Pre-compute 1d volume spike filter
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Pre-compute 1d ATR(14) for stoploss
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current high
    prev_low[0] = low_1d[0]    # first bar uses current low
    prev_close[0] = close_1d[0] # first bar uses current close
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.1 * camarilla_range / 2
    l3 = prev_close - 1.1 * camarilla_range / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_slope_aligned[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(h3[i]) or np.isnan(l3[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L3 (mean reversion) OR stoploss hit
            if close_1d[i] <= l3[i] or close_1d[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H3 (mean reversion) OR stoploss hit
            if close_1d[i] >= h3[i] or close_1d[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla reversals with volume and trend filters
            if vol_spike[i]:
                # Long: price rejects L3 and closes above it with uptrend
                if close_1d[i] <= l3[i] * 1.005 and ema_21_slope_aligned[i] > 0:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: price rejects H3 and closes below it with downtrend
                elif close_1d[i] >= h3[i] * 0.995 and ema_21_slope_aligned[i] < 0:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals