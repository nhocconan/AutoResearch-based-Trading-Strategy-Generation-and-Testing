#!/usr/bin/env python3
"""
Experiment #004: 4h HMA Trend + Volatility Expansion + Range Breakout

Hypothesis: Previous regime-switching strategies failed due to complexity and whipsaw.
This strategy uses a simpler, more robust approach proven in crypto:

1. 12h HMA(21) Trend Filter - Clean trend direction via mtf_data helper.
   Only long when price > 12h HMA, only short when price < 12h HMA.
   HMA has less lag than EMA, better for crypto's fast moves.

2. ATR Expansion Ratio (7/21) - Detects volatility expansion before big moves.
   Entry when ATR(7)/ATR(21) > 1.3 (vol expanding). Avoids low-vol chop.

3. Price Position in 55-bar Range - Where price sits in recent high-low range.
   Long when > 60% of range, short when < 40% of range.
   Simpler than Donchian breakout, catches continuation better.

4. RSI(14) Momentum Confirmation - RSI > 52 for longs, RSI < 48 for shorts.
   Mild thresholds to ensure trade frequency (avoid 0-trade failure).

5. Asymmetric Entry Thresholds - Crypto has long bias. Longs need 3 conditions,
   shorts need 4 conditions (harder to short = fewer short whipsaws).

6. 2.5x ATR Trailing Stop - Protects against 2022-style crashes.
   Signal → 0 when stopped out.

7. Conservative Sizing - 0.20 base, 0.30 high conviction (12h + 1d aligned).
   Discrete levels minimize fee churn.

Why this should beat Sharpe=0.278:
- Simpler logic = more trades (avoids 0-trade failure mode)
- Vol expansion filter catches real breakouts, not fakeouts
- Asymmetric thresholds account for crypto's long bias
- HMA trend filter is proven (current best uses HMA)
- 4h timeframe targets 20-50 trades/year (optimal fee drag)

Timeframe: 4h (REQUIRED)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_vol_expand_range_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_range_position(high, low, close, period=55):
    """
    Calculate where price sits in recent high-low range.
    Returns value 0.0 to 1.0 (0 = at range low, 1 = at range high)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    range_high = high_s.rolling(window=period, min_periods=period).max()
    range_low = low_s.rolling(window=period, min_periods=period).min()
    range_size = range_high - range_low
    
    # Avoid division by zero
    range_size = range_size.replace(0, np.nan)
    
    position = (close - range_low) / range_size
    position = position.replace([np.inf, -np.inf], np.nan)
    
    return position.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend filter
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_21 = calculate_atr(high, low, close, 21)
    atr_14 = calculate_atr(high, low, close, 14)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    range_pos = calculate_range_position(high, low, close, period=55)
    
    # ATR expansion ratio
    atr_ratio = atr_7 / atr_21
    atr_ratio = np.where(atr_21 > 0, atr_ratio, np.nan)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.20
    HIGH_CONV_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(range_pos[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === 12H TREND FILTER ===
        twelve_h_bullish = close[i] > hma_12h_21_aligned[i]
        twelve_h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === VOLATILITY EXPANSION ===
        vol_expanding = atr_ratio[i] > 1.3
        vol_strong = atr_ratio[i] > 1.5
        
        # === RANGE POSITION ===
        range_high_position = range_pos[i] > 0.60
        range_low_position = range_pos[i] < 0.40
        range_extreme_high = range_pos[i] > 0.75
        range_extreme_low = range_pos[i] < 0.25
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 52
        rsi_bearish = rsi_14[i] < 48
        rsi_strong_bull = rsi_14[i] > 58
        rsi_strong_bear = rsi_14[i] < 42
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY (asymmetric - easier to enter long in crypto)
        # Need: trend + (vol expansion OR range breakout) + RSI
        long_score = 0
        
        if twelve_h_bullish:
            long_score += 2  # Trend alignment (required)
        
        if vol_expanding:
            long_score += 1.5
        if vol_strong:
            long_score += 0.5  # Extra for strong vol
        
        if range_high_position:
            long_score += 1
        if range_extreme_high:
            long_score += 0.5
        
        if rsi_bullish:
            long_score += 1
        if rsi_strong_bull:
            long_score += 0.5
        
        # Long entry threshold: 4.0 (moderate for trade frequency)
        if long_score >= 4.0:
            if twelve_h_bullish and vol_strong:
                new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction
            else:
                new_signal = BASE_SIZE  # 0.20 - base
        
        # SHORT ENTRY (asymmetric - harder to enter short)
        # Need: trend + vol expansion + range breakdown + RSI
        short_score = 0
        
        if twelve_h_bearish:
            short_score += 2  # Trend alignment (required)
        
        if vol_expanding:
            short_score += 1.5
        if vol_strong:
            short_score += 0.5
        
        if range_low_position:
            short_score += 1.5  # Higher weight for shorts
        if range_extreme_low:
            short_score += 0.5
        
        if rsi_bearish:
            short_score += 1
        if rsi_strong_bear:
            short_score += 0.5
        
        # Short entry threshold: 5.0 (harder than longs)
        if short_score >= 5.0:
            if twelve_h_bearish and vol_strong:
                new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
            else:
                new_signal = -BASE_SIZE  # -0.20 - base
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~240 hours = 10 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            # Weaker long entry
            if twelve_h_bullish and range_high_position and rsi_bullish:
                new_signal = BASE_SIZE
            # Weaker short entry (still harder)
            elif twelve_h_bearish and range_low_position and rsi_bearish and vol_expanding:
                new_signal = -BASE_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend turns bearish
            if position_side > 0 and twelve_h_bearish:
                trend_reversal = True
            # Exit short if 12h trend turns bullish
            if position_side < 0 and twelve_h_bullish:
                trend_reversal = True
        
        # === RANGE REVERSAL EXIT ===
        range_exit = False
        if in_position and position_side != 0:
            # Exit long if price falls back to middle of range
            if position_side > 0 and range_pos[i] < 0.45:
                range_exit = True
            # Exit short if price rises back to middle of range
            if position_side < 0 and range_pos[i] > 0.55:
                range_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or range_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals