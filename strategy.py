#!/usr/bin/env python3
"""
Experiment #527: 1d Primary + 1w HTF — KAMA Adaptive Trend + ADX Regime + RSI Pullback

Hypothesis: After 472 failed strategies (mostly complex volspike/choppiness/connors combos),
return to SIMPLER adaptive trend-following with fewer conflicting filters.

Key insights from failures:
- Volatility spike strategies: ALL failed (volspike_* all discarded)
- Choppiness Index: Failed repeatedly (chop_* all negative Sharpe)
- Connors RSI: Failed on BTC/ETH (only worked on SOL)
- Complex multi-condition entries = 0 trades or negative Sharpe
- 1d timeframe shows promise (current best Sharpe=0.435)

This strategy uses:
1. KAMA(21) adaptive MA - adapts to volatility, less whipsaw than HMA/EMA
2. 1w KAMA(21) for major trend regime - only trade with weekly trend
3. ADX(14) for trend strength - only enter when ADX > 18 (trending)
4. RSI(14) pullback entries - enter on pullbacks in direction of trend
5. Asymmetric position sizing - 0.35 in strong trends (ADX>25), 0.20 otherwise
6. ATR(14) 2.5x trailing stop for risk management

Why this might work:
- KAMA adapts to market regime (fast in trends, slow in chop) - research note #3
- 1w trend filter prevents counter-trend trades (major failure mode in 2022)
- ADX filter avoids trading in choppy markets (whipsaw destruction)
- RSI pullback = better entry timing than breakout chasing
- Simple logic = consistent signals across BTC/ETH/SOL
- 1d TF targets 20-40 trades/year (optimal fee/trade ratio per Rule 10)

Position sizing: 0.20-0.35 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_rsi_pullback_1w_v1"
timeframe = "1d"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - fast in trends, slow in chop.
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    volatility[0:period] = np.nan
    er = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate SC (smoothing constant)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = strong trend, ADX < 20 = choppy/range
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth using Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF KAMA for major trend regime
    kama_1w_21 = calculate_kama(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_1d_21 = calculate_kama(close, period=21)
    kama_1d_50 = calculate_kama(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_STRONG = 0.35  # ADX > 25
    POSITION_SIZE_WEAK = 0.20    # ADX 18-25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track KAMA crossover
    prev_kama_21 = np.zeros(n)
    prev_kama_50 = np.zeros(n)
    prev_kama_21[1:] = kama_1d_21[:-1]
    prev_kama_50[1:] = kama_1d_50[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1w_21_aligned[i]):
            continue
        if np.isnan(kama_1d_21[i]) or np.isnan(kama_1d_50[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        bull_regime = close[i] > kama_1w_21_aligned[i]
        bear_regime = close[i] < kama_1w_21_aligned[i]
        
        # === 1D KAMA TREND ===
        kama_bull = kama_1d_21[i] > kama_1d_50[i]
        kama_bear = kama_1d_21[i] < kama_1d_50[i]
        
        # KAMA crossover signals
        kama_cross_up = (kama_1d_21[i] > kama_1d_50[i]) and (prev_kama_21[i] <= prev_kama_50[i])
        kama_cross_down = (kama_1d_21[i] < kama_1d_50[i]) and (prev_kama_21[i] >= prev_kama_50[i])
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] > 18.0
        no_trend = adx_14[i] <= 18.0
        
        # === RSI PULLBACK FILTER ===
        rsi_pullback_long = rsi_14[i] < 55.0  # Pullback in uptrend
        rsi_pullback_short = rsi_14[i] > 45.0  # Pullback in downtrend
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_long = rsi_14[i] < 30.0
        rsi_extreme_short = rsi_14[i] > 70.0
        
        # === POSITION SIZING BASED ON ADX ===
        if strong_trend:
            pos_size = POSITION_SIZE_STRONG
        elif weak_trend:
            pos_size = POSITION_SIZE_WEAK
        else:
            pos_size = 0.0  # No trade in chop
        
        # === ENTRY LOGIC — SIMPLE WITH CONFLUENCE ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if bull_regime and kama_bull and weak_trend:
            # Condition 1: KAMA crossover up + bull regime
            if kama_cross_up:
                new_signal = pos_size
            # Condition 2: KAMA aligned + RSI pullback (not extreme)
            elif rsi_pullback_long and not rsi_extreme_long:
                new_signal = pos_size * 0.8
            # Condition 3: RSI oversold in bull regime (mean reversion)
            elif rsi_oversold:
                new_signal = pos_size
        
        # SHORT ENTRIES (only if no long signal)
        if new_signal == 0.0 and bear_regime and kama_bear and weak_trend:
            # Condition 1: KAMA crossover down + bear regime
            if kama_cross_down:
                new_signal = -pos_size
            # Condition 2: KAMA aligned + RSI pullback (not extreme)
            elif rsi_pullback_short and not rsi_extreme_short:
                new_signal = -pos_size * 0.8
            # Condition 3: RSI overbought in bear regime (mean reversion)
            elif rsi_overbought:
                new_signal = -pos_size
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or ADX collapse) ===
        # Exit long on regime flip to bear or trend weakness
        if in_position and position_side > 0:
            if bear_regime and kama_bear:
                new_signal = 0.0
            elif no_trend and adx_14[i] < 15.0:  # ADX collapsing
                new_signal = 0.0
            elif rsi_14[i] > 75.0:  # Extreme overbought
                new_signal = 0.0
        
        # Exit short on regime flip to bull or trend weakness
        if in_position and position_side < 0:
            if bull_regime and kama_bull:
                new_signal = 0.0
            elif no_trend and adx_14[i] < 15.0:  # ADX collapsing
                new_signal = 0.0
            elif rsi_14[i] < 25.0:  # Extreme oversold
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals