#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly volume confirmation and ATR-based stoploss
# - Long: KAMA rising (bullish trend) + weekly volume > 1.5x 20-week average + price > KAMA
# - Short: KAMA falling (bearish trend) + weekly volume > 1.5x 20-week average + price < KAMA
# - Exit: ATR-based trailing stop (2.0 ATR from extreme) or opposite KAMA signal
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - KAMA adapts to market noise, reducing whipsaws in choppy markets
# - Weekly volume confirmation filters out low-conviction moves
# - ATR stoploss manages risk during volatile periods
# - Works in both bull (trend following) and bear (trend following with stops) markets

name = "1d_1w_kama_volume_trend_v1"
timeframe = "1d"
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
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load weekly data ONCE before loop for volume confirmation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute weekly volume confirmation (20-period average)
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute KAMA on daily timeframe
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # Already on daily timeframe
    
    # Pre-compute ATR for stoploss (daily timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        kama_val = kama_aligned[i]
        
        # Weekly volume confirmation: current volume > 1.5x 20-week average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_val > kama_aligned[i-1]
        kama_falling = kama_val < kama_aligned[i-1]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA rising + volume confirmation + price above KAMA
        if kama_rising and vol_confirm and close_price > kama_val:
            enter_long = True
        
        # Short: KAMA falling + volume confirmation + price below KAMA
        if kama_falling and vol_confirm and close_price < kama_val:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or KAMA turns falling
            exit_long = (close_price <= long_stop) or kama_falling
        elif position == -1:
            # Exit short if price hits ATR stoploss or KAMA turns rising
            exit_short = (close_price >= short_stop) or kama_rising
        
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
            long_stop = max(long_stop, high[i] - 2.0 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2*ATR)
            short_stop = min(short_stop, low[i] + 2.0 * atr_14[i])
        
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