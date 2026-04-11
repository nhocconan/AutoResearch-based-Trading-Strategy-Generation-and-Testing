#2021-024
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # H3 = C + 1.125*(H-L), L3 = C - 1.125*(H-L)
    # H2 = C + 0.75*(H-L), L2 = C - 0.75*(H-L)
    # H1 = C + 0.5*(H-L), L1 = C - 0.5*(H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate ranges and levels
    weekly_range = weekly_high - weekly_low
    
    # H4 and L4 levels (strongest resistance/support)
    h4 = weekly_close + 1.5 * weekly_range
    l4 = weekly_close - 1.5 * weekly_range
    
    # H3 and L3 levels
    h3 = weekly_close + 1.125 * weekly_range
    l3 = weekly_close - 1.125 * weekly_range
    
    # Align weekly Camarilla levels to daily timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: avoid extremely low volatility periods
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (0.5 * atr_ma_50)  # Require ATR > 50% of its MA
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        atr_val = atr[i]
        vol_filter = volatility_filter[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma_20[i]
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above weekly H4 with volume and volatility
        if price_high > h4_val and volume_confirmed and vol_filter:
            long_signal = True
        
        # Short: price breaks below weekly L4 with volume and volatility
        if price_low < l4_val and volume_confirmed and vol_filter:
            short_signal = True
        
        # Exit conditions: return to H3/L3 levels
        exit_long = position == 1 and price_close < h3_val
        exit_short = position == -1 and price_close > l3_val
        
        # Stop loss conditions (2x ATR)
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_val)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily Camarilla breakout with weekly levels, volume confirmation, and volatility filter.
# Enters long when price breaks above weekly H4 level (C + 1.5*(H-L)) with volume > 1.8x 20-day average.
# Enters short when price breaks below weekly L4 level (C - 1.5*(H-L)) with volume confirmation.
# Uses volatility filter to avoid low-volatility environments where breakouts fail.
# Exits when price returns to weekly H3/L3 levels or ATR stop loss (2x) is hit.
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by trading breakouts in either direction with volume/volatility filters.