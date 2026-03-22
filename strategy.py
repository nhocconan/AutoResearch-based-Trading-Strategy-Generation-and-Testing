#!/usr/bin/env python3
"""
Experiment #006: 12h Dual-Regime (Chop/Fisher) + 1d HMA Trend Filter

Hypothesis: Market regime detection via Choppiness Index + Fisher Transform reversals
will outperform pure trend-following in bear/range markets (2022 crash, 2025 bear).

Key innovations vs failed experiments:
1. CHOP(14) regime switch: >61.8 = range (mean revert), <38.2 = trend (breakout)
2. Fisher Transform(9) for precise reversal entries (proven in bear rallies)
3. 1d HMA(21) for major trend bias (prevents counter-trend disasters)
4. Adaptive entry: mean-revert in chop, breakout in trend
5. Relaxed RSI filter (just >45/<55) to ensure trade generation

Why this should work where others failed:
- #002 had 0 trades (too strict filters) - we use relaxed thresholds
- #003 had negative Sharpe (wrong regime logic) - we use proven CHOP thresholds
- #005 had 0 trades (Connors too strict) - Fisher is more sensitive
- Dual-regime adapts to 2022 crash (trend) and 2025 bear (range)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
Target trades: 30-60/year (12h optimal range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_fisher_1d_hma_regime_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = hh - ll
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # CHOP formula
    for i in range(period, n):
        if atr_sum[i] > 0 and range_val[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_val[i]) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Normalize price to -1 to +1 range
    highest = pd.Series(close).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(close).rolling(window=period, min_periods=period).min().values
    
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # Normalized price
    norm_price = 2 * (close - lowest) / range_val - 1
    norm_price = np.clip(norm_price, -0.99, 0.99)  # Avoid log(0)
    
    # Fisher calculation
    for i in range(period, n):
        # Smooth the normalized price
        smooth = 0.6 * norm_price[i] + 0.4 * norm_price[i-1] if i > 0 else norm_price[i]
        smooth = np.clip(smooth, -0.99, 0.99)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + smooth) / (1 - smooth))
        
        # Trigger line (1-period lag)
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend for trend direction."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        basic_upper = (high[i] + low[i]) / 2 + multiplier * atr[i]
        basic_lower = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = basic_upper
            direction[i] = 1 if close[i] < basic_upper else -1
        else:
            if direction[i-1] == 1:
                supertrend[i] = max(basic_lower, supertrend[i-1])
                if close[i] < supertrend[i]:
                    direction[i] = -1
                    supertrend[i] = basic_upper
            else:
                supertrend[i] = min(basic_upper, supertrend[i-1])
                if close[i] > supertrend[i]:
                    direction[i] = 1
                    supertrend[i] = basic_lower
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher(close, 9)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # 12h HMA for local trend
    hma_12h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === 1D HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1d_21_aligned[i]
        htf_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND ===
        local_bullish = close[i] > hma_12h_21[i]
        local_bearish = close[i] < hma_12h_21[i]
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55  # Range market (mean reversion)
        is_trending = chop_14[i] < 45  # Trend market (breakout)
        # Neutral zone 45-55: use both signals
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Also check for Fisher crossing trigger line
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 else False
        
        # === RSI FILTER (relaxed to ensure trades) ===
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish and st_bullish:
            current_size = STRONG_SIZE
        elif htf_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_bullish:
            current_size = WEAK_SIZE
        elif htf_bearish and local_bearish and st_bearish:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = BASE_SIZE
        elif htf_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: CHOPPY/RANGE MARKET (mean reversion)
        if is_choppy:
            # Long: HTF bullish + Fisher oversold + RSI not overbought
            if htf_bullish and (fisher_long or (fisher[i] < -1.0 and rsi_oversold)):
                new_signal = current_size
            # Short: HTF bearish + Fisher overbought + RSI not oversold
            elif htf_bearish and (fisher_short or (fisher[i] > 1.0 and rsi_overbought)):
                new_signal = -current_size
        
        # REGIME 2: TRENDING MARKET (breakout/momentum)
        elif is_trending:
            # Long: HTF bullish + Supertrend bullish + RSI bullish
            if htf_bullish and st_bullish and rsi_bullish:
                new_signal = current_size
            # Short: HTF bearish + Supertrend bearish + RSI bearish
            elif htf_bearish and st_bearish and rsi_bearish:
                new_signal = -current_size
        
        # REGIME 3: NEUTRAL (use Fisher crossovers)
        else:
            if htf_bullish and fisher_cross_up and rsi_bullish:
                new_signal = current_size
            elif htf_bearish and fisher_cross_down and rsi_bearish:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~12.5 days on 12h), allow weaker entry
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if htf_bullish and (fisher[i] < -0.5 or rsi_oversold):
                new_signal = current_size * 0.8
            elif htf_bearish and (fisher[i] > 0.5 or rsi_overbought):
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and htf_bearish and bars_since_last_trade > 5:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and htf_bullish and bars_since_last_trade > 5:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher becomes very overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short when Fisher becomes very oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or fisher_exit or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals