#!/usr/bin/env python3
"""
Experiment #179: 12h Fisher Transform Reversals with Daily HMA Bias
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025).
Combined with Daily HMA for major trend bias and volume confirmation for entry quality.
12h timeframe reduces noise and fee impact while capturing multi-day swings.
Entry: Fisher crosses -1.5 (long) or +1.5 (short) with Daily HMA confirmation.
Stoploss: 2.5*ATR trailing. Position size: 0.25 discrete levels.
This targets reversals in 2022 crash and 2025 consolidation where trend-following failed.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_daily_hma_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5.
    Reference: John Ehlers, "Rocket Science for Traders"
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate effective price (weighted average of HL2)
    ema_hl2 = hl2_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Normalize to -1 to +1 range
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    range_hl = highest - lowest
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    normalized = (2.0 * (hl2 - lowest) / range_hl) - 1.0
    normalized = np.clip(normalized, -0.99, 0.99)  # Prevent log errors
    
    # Fisher calculation
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.where(np.isnan(fisher), 0.0, fisher)
    
    # Signal line (1-period lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (buying pressure)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    ratio = np.clip(ratio, 0, 1)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 12h trend
        trend_bullish = hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i]
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_buying = vol_ratio[i] > 0.55
        vol_selling = vol_ratio[i] < 0.45
        
        # RSI filter (avoid extreme overbought/oversold for reversals)
        rsi_neutral = 35 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === FISHER REVERSAL LONG ===
        # Entry when Fisher crosses above -1.5 (oversold reversal)
        if fisher_long:
            # Require daily trend not bearish OR strong volume buying
            if not daily_bearish or vol_buying:
                # Additional confirmation: RSI rising or volume buying
                if rsi[i] > rsi[i-2] if i > 2 else True:
                    new_signal = SIZE_ENTRY
        
        # === FISHER REVERSAL SHORT ===
        # Entry when Fisher crosses below +1.5 (overbought reversal)
        elif fisher_short:
            # Require daily trend not bullish OR strong volume selling
            if not daily_bullish or vol_selling:
                # Additional confirmation: RSI falling or volume selling
                if rsi[i] < rsi[i-2] if i > 2 else True:
                    new_signal = -SIZE_ENTRY
        
        # === TREND CONTINUATION (when Fisher confirms trend) ===
        if new_signal == 0.0:
            # Long continuation: bullish trend + Fisher rising from neutral
            if trend_bullish and daily_bullish and fisher_rising:
                if -1.0 < fisher[i] < 0.0:  # Fisher in neutral-positive zone
                    if vol_buying:
                        new_signal = SIZE_ENTRY
            
            # Short continuation: bearish trend + Fisher falling from neutral
            elif trend_bearish and daily_bearish and fisher_falling:
                if 0.0 < fisher[i] < 1.0:  # Fisher in neutral-negative zone
                    if vol_selling:
                        new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals