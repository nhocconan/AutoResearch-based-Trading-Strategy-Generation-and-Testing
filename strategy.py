#!/usr/bin/env python3
"""
Experiment #004: 4h KAMA Trend + 12h/1d HMA Filters + ADX Confirmation

Hypothesis: Previous strategies failed because CRSI and Choppiness Index created
too many whipsaws in crypto's trending nature. This strategy uses:

1. KAMA(10,2,30) on 4h - Kaufman Adaptive MA adjusts to volatility, reduces
   whipsaws in choppy markets while capturing trends. Proven in exp #003.

2. 12h HMA(21) - Intermediate trend filter via mtf_data helper. Only trade
   in direction of 12h trend. Reduces counter-trend failures.

3. 1d HMA(21) - Major trend bias via mtf_data helper. Increases position size
   when 4h, 12h, and 1d all align (high conviction trades).

4. ADX(14) > 20 - Confirms trend strength. Avoids entering in dead markets.
   ADX > 25 for high conviction entries.

5. ATR(14) Trailing Stop - 2.5x ATR for risk management. Signal → 0 when stopped.

6. Simple Entry Logic - KAMA crossover + ADX + HTF alignment. Fewer conditions
   = more trades (avoids 0-trade failure from exp #001/#002).

Why this should work:
- KAMA worked in exp #003 (Sharpe=0.096) - build on success
- 12h/1d HTF filters prevent false breakouts (learned from exp #001 failure)
- 4h timeframe = 20-50 trades/year target (optimal for fee drag)
- Conservative sizing (0.25-0.30) protects against crashes
- Simpler logic than CRSI/Chop = more trades generated

Timeframe: 4h (REQUIRED for Experiment #004)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction (3+ TF alignment)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_hma_adx_12h_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adjusts smoothing based on market noise.
    ER (Efficiency Ratio) = |Price Change| / Sum of |Individual Changes|
    SC (Smoothing Constant) = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close_s - close_s.shift(efficiency_period))
    noise = np.abs(close_s - close_s.shift(1))
    noise_sum = noise.rolling(window=efficiency_period, min_periods=efficiency_period).sum()
    
    er = price_change / noise_sum.replace(0, np.inf)
    er = er.replace([np.inf, -np.inf], np.nan)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[efficiency_period] = close_s.iloc[efficiency_period]
    
    for i in range(efficiency_period + 1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    adx = adx.replace([np.inf, -np.inf], np.nan)
    
    return adx.values, plus_di.values, minus_di.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for intermediate trend filter
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # KAMA previous value for crossover detection
    kama_prev = np.roll(kama_4h, 1)
    kama_prev[:10] = np.nan  # First 10 values invalid
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(kama_prev[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            continue
        
        # === 12H INTERMEDIATE TREND FILTER ===
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 1D MAJOR TREND BIAS ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # KAMA crossover signals
        kama_cross_long = (close[i] > kama_4h[i]) and (close[i-1] <= kama_prev[i-1])
        kama_cross_short = (close[i] < kama_4h[i]) and (close[i-1] >= kama_prev[i-1])
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25
        adx_moderate = adx_14[i] > 20
        
        # DI confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        long_score = 0
        
        # KAMA position (must be above KAMA)
        if kama_bullish:
            long_score += 2
        
        # KAMA crossover (stronger signal)
        if kama_cross_long:
            long_score += 2
        
        # 12h trend alignment
        if trend_12h_bullish:
            long_score += 2
        
        # 1d major bias
        if trend_1d_bullish:
            long_score += 1
        
        # ADX confirms trend
        if adx_moderate:
            long_score += 1
        if adx_strong:
            long_score += 1
        
        # DI confirmation
        if di_bullish:
            long_score += 1
        
        # Enter long if score >= 6 (moderate threshold for trade frequency)
        if long_score >= 6:
            # Determine position size based on conviction
            if trend_1d_bullish and trend_12h_bullish and adx_strong:
                new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction (3+ TF align)
            elif trend_12h_bullish and adx_moderate:
                new_signal = BASE_SIZE  # 0.25 - base
            else:
                new_signal = LOW_CONV_SIZE  # 0.15 - low conviction
        
        # SHORT ENTRY
        short_score = 0
        
        # KAMA position (must be below KAMA)
        if kama_bearish:
            short_score += 2
        
        # KAMA crossover (stronger signal)
        if kama_cross_short:
            short_score += 2
        
        # 12h trend alignment
        if trend_12h_bearish:
            short_score += 2
        
        # 1d major bias
        if trend_1d_bearish:
            short_score += 1
        
        # ADX confirms trend
        if adx_moderate:
            short_score += 1
        if adx_strong:
            short_score += 1
        
        # DI confirmation
        if di_bearish:
            short_score += 1
        
        # Enter short if score >= 6
        if short_score >= 6:
            # Determine position size based on conviction
            if trend_1d_bearish and trend_12h_bearish and adx_strong:
                new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
            elif trend_12h_bearish and adx_moderate:
                new_signal = -BASE_SIZE  # -0.25 - base
            else:
                new_signal = -LOW_CONV_SIZE  # -0.15 - low conviction
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~240 hours = 10 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if kama_bullish and trend_12h_bullish and di_bullish:
                new_signal = LOW_CONV_SIZE
            elif kama_bearish and trend_12h_bearish and di_bearish:
                new_signal = -LOW_CONV_SIZE
        
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
            if position_side > 0 and trend_12h_bearish:
                trend_reversal = True
            # Exit short if 12h trend turns bullish
            if position_side < 0 and trend_12h_bullish:
                trend_reversal = True
        
        # === KAMA REVERSAL EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Exit long if price crosses below KAMA
            if position_side > 0 and close[i] < kama_4h[i]:
                kama_exit = True
            # Exit short if price crosses above KAMA
            if position_side < 0 and close[i] > kama_4h[i]:
                kama_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or kama_exit:
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