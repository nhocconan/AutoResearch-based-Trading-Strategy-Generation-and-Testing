#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long: Price breaks above Camarilla H3 level (1d) + 1w close > 1w open (bullish trend) + volume > 1.5x 20-period average
# - Short: Price breaks below Camarilla L3 level (1d) + 1w close < 1w open (bearish trend) + volume > 1.5x 20-period average
# - Exit: Opposite Camarilla level (L3 for long, H3 for short) or ATR-based stoploss (2.0 ATR)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Camarilla pivots provide mathematically derived support/resistance levels that work in ranging and trending markets
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out weak breakouts and increases signal quality

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 for breakouts
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed as they're based on completed prior day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w trend filter (bullish if close > open, bearish if close < open)
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    trend_1w_bullish = close_1w > open_1w  # True for bullish weekly candle
    trend_1w_bearish = close_1w < open_1w  # True for bearish weekly candle
    
    # Align 1w trend to 1d timeframe (additional_delay_bars=1 to wait for weekly close)
    trend_1w_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_bullish.astype(float), additional_delay_bars=1)
    trend_1w_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_bearish.astype(float), additional_delay_bars=1)
    
    # Pre-compute ATR for stoploss (1d timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(trend_1w_bullish_aligned[i]) or np.isnan(trend_1w_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # Camarilla levels
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Trend filter
        is_bullish_trend = trend_1w_bullish_aligned[i] > 0.5
        is_bearish_trend = trend_1w_bearish_aligned[i] > 0.5
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above H3 with volume confirmation and bullish weekly trend
        if close_price > h3_level and vol_confirm and is_bullish_trend:
            enter_long = True
        
        # Short breakout: price breaks below L3 with volume confirmation and bearish weekly trend
        if close_price < l3_level and vol_confirm and is_bearish_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 or hits ATR stoploss
            exit_long = (close_price < l3_level) or (close_price <= long_stop)
        elif position == -1:
            # Exit short if price breaks above H3 or hits ATR stoploss
            exit_short = (close_price > h3_level) or (close_price >= short_stop)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.0 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.0 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2*ATR)
            long_stop = max(long_stop, high_price - 2.0 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2*ATR)
            short_stop = min(short_stop, low_price + 2.0 * atr_14[i])
        
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