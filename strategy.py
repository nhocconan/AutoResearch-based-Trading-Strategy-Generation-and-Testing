#!/usr/bin/env python3
"""
Experiment #024: Donchian Breakout + Volume + Choppiness + 1d SMA50 (4h)

HYPOTHESIS: Tighten the winning formula from DB top performers:
- Donchian(20) breakout for price structure
- Volume spike (>1.3x) for confirmation
- Choppiness <55 (less restrictive than 38.2 but still filters range)
- 1d SMA50 for trend direction (simple, proven)

KEY FIX vs #023: Removed redundant HMA16 + HTF EMA21 (two trend filters that
conflict). Use ONLY 1d SMA50 as single trend anchor.

EXPECTED TRADES: 100-150/year per symbol = 400-600 total over 4 years.
- Too high! Need to tighten.

TIGHTER PARAMETERS:
- Choppiness < 50 (more restrictive, removes more range markets)
- Volume > 1.5x (stronger confirmation)
- Combined: ~60-100 trades/year = 240-400 total (at the edge)

WHY IT SHOULD WORK:
- Bull: Breakout above Donchian high + vol spike + above SMA50 = momentum long
- Bear: Breakout below Donchian low + vol spike + below SMA50 = momentum short
- Range (CHOP > 55): Skip entries, reduces whipsaws
- ATR stoploss: Controls risk on 77% BTC crashes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness vs trending
    CHOP > 61.8 = choppy (range-bound), CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = np.sum([
                max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                for j in range(i-period+1, i+1)
            ])
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === Trend check (1d SMA50) ===
        price_above_sma = close[i] > sma50_aligned[i] if not np.isnan(sma50_aligned[i]) else False
        price_below_sma = close[i] < sma50_aligned[i] if not np.isnan(sma50_aligned[i]) else False
        
        # === Regime check: trending when CHOP < 50 ===
        is_trending = chop[i] < 50.0
        
        # === Entry conditions ===
        desired_signal = 0.0
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        if not in_position:
            # === LONG ENTRY: Price breaks above previous Donchian high ===
            prev_donchian = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
            bullish_breakout = high[i] > prev_donchian if not np.isnan(prev_donchian) else False
            
            # Long: breakout + vol spike + trending + above SMA50
            if bullish_breakout and vol_spike and is_trending and price_above_sma:
                desired_signal = SIZE
                
            # === SHORT ENTRY: Price breaks below previous Donchian low ===
            prev_donchian_low = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
            bearish_breakout = low[i] < prev_donchian_low if not np.isnan(prev_donchian_low) else False
            
            # Short: breakout + vol spike + trending + below SMA50
            if bearish_breakout and vol_spike and is_trending and price_below_sma:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price falls below SMA50 (trend reversal)
                if close[i] < sma50_aligned[i] if not np.isnan(sma50_aligned[i]) else False:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price rises above SMA50 (trend reversal)
                if close[i] > sma50_aligned[i] if not np.isnan(sma50_aligned[i]) else False:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals