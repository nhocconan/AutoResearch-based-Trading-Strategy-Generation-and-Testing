#!/usr/bin/env python3
"""
Experiment #1083: 1d Primary + 1w HTF — Dual Regime HMA/RSI/Donchian

Hypothesis: Daily timeframe with weekly macro filter can work IF we simplify entry logic.
Previous 1d strategies failed because CRSI/KAMA were too complex and filters conflicted.

New approach:
1. DUAL REGIME via Choppiness Index (CHOP):
   - CHOP > 61.8 = Range → Mean reversion (RSI extremes + Bollinger bands)
   - CHOP < 38.2 = Trend → Breakout following (Donchian + HMA alignment)
2. 1w HMA21 for macro bias — only trade in direction of weekly trend
3. HMA(21) on 1d for intermediate trend confirmation
4. RSI(14) for entry timing — simpler than CRSI, proven on daily charts
5. Donchian(20) for breakout confirmation in trending regime
6. ATR(14) trailing stop at 2.5x for risk management

Why this should beat Sharpe=0.612:
- Simpler logic = more trades triggered (previous 1d strategies had 0 trades)
- Dual regime adapts to market conditions (range vs trend)
- 1d timeframe = 20-50 trades/year target (optimal for fee/trade balance)
- Weekly HTF provides macro filter without overcomplicating
- Different from all 784 failed strategies (no CRSI, no Fisher, no STC)

Timeframe: 1d (primary)
HTF: 1w (weekly) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_hma_rsi_donchian_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    CHOP > 61.8 = choppy/range market (mean reversion favored)
    CHOP < 38.2 = trending market (breakout/trend follow favored)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, 21)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA21) ===
        trend_bull = close[i] > hma_1d[i]
        trend_bear = close[i] < hma_1d[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # === BOLLINGER TOUCH ===
        bb_touch_lower = low[i] <= bb_lower[i] * 1.002
        bb_touch_upper = high[i] >= bb_upper[i] * 0.998
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === CHOPPY REGIME: MEAN REVERSION ===
        if is_choppy:
            # Long: RSI oversold + price at BB lower + macro bullish bias
            if rsi_oversold and bb_touch_lower and macro_bull:
                desired_signal = current_size
            # Long: RSI extreme oversold + macro bullish (stronger signal)
            elif rsi_extreme_oversold and macro_bull and trend_bull:
                desired_signal = current_size
            
            # Short: RSI overbought + price at BB upper + macro bearish bias
            elif rsi_overbought and bb_touch_upper and macro_bear:
                desired_signal = -current_size
            # Short: RSI extreme overbought + macro bearish (stronger signal)
            elif rsi_extreme_overbought and macro_bear and trend_bear:
                desired_signal = -current_size
        
        # === TRENDING REGIME: BREAKOUT FOLLOWING ===
        elif is_trending:
            # Long: Donchian breakout + RSI bullish + macro + trend aligned
            if donch_breakout_long and rsi[i] > 50.0 and macro_bull and trend_bull:
                desired_signal = current_size
            # Long: Donchian breakout + macro bullish (weaker)
            elif donch_breakout_long and macro_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: Donchian breakout + RSI bearish + macro + trend aligned
            elif donch_breakout_short and rsi[i] < 50.0 and macro_bear and trend_bear:
                desired_signal = -current_size
            # Short: Donchian breakout + macro bearish (weaker)
            elif donch_breakout_short and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TRANSITION ZONE (38.2 <= CHOP <= 61.8) ===
        else:
            # Require stronger confluence in transition zone
            # Long: RSI oversold + trend bullish + macro bullish
            if rsi_oversold and trend_bull and macro_bull:
                desired_signal = REDUCED_SIZE
            # Short: RSI overbought + trend bearish + macro bearish
            elif rsi_overbought and trend_bear and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish or RSI > 40
                if trend_bull or rsi[i] > 40.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if trend still bearish or RSI < 60
                if trend_bear or rsi[i] < 60.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if RSI overbought + price at BB upper
            if rsi_overbought and bb_touch_upper:
                desired_signal = 0.0
            # Exit long if macro reverses bearish + trend reverses
            if macro_bear and trend_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if RSI oversold + price at BB lower
            if rsi_oversold and bb_touch_lower:
                desired_signal = 0.0
            # Exit short if macro reverses bullish + trend reverses
            if macro_bull and trend_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals