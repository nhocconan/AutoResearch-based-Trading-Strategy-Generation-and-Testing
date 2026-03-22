#!/usr/bin/env python3
"""
Experiment #331: 4h Primary + 1d/1w HTF — Dual Regime (Trend/Mean Revert) + KAMA + RSI

Hypothesis: A dual-regime approach using Choppiness Index to switch between trend-following
and mean-reversion modes will outperform single-mode strategies on 4h timeframe.

Why this might work (learning from 300+ failed experiments):
1. Choppiness Index (CHOP) reliably identifies regime: CHOP<45=trend, CHOP>55=range
2. KAMA adapts to volatility better than HMA/EMA - smooth in chop, responsive in trends
3. 1d HTF KAMA provides major trend bias without over-filtering (avoid 0-trade problem)
4. Simpler entry conditions = more trades generated (critical lesson from failures #320,#324,#325,#328,#330)
5. Asymmetric sizing (longs 0.30-0.35, shorts 0.20-0.25) matches crypto's long bias
6. Target: 25-50 trades/year on 4h (appropriate frequency, low fee drag)

Key differences from failed 4h strategies:
- Fewer conflicting filters (max 3 conditions per entry, not 5+)
- Regime determines ENTRY TYPE, not whether to trade at all
- 1d HTF for bias only, not hard filter (avoid regime flip killing all trades)
- RSI ranges widened (35-65 instead of 30-70) to generate more signals
- Frequency safeguard: force trade after 20 bars of no activity

Position sizing: 0.30 base, 0.35 strong conviction (longs), 0.20-0.25 (shorts)
Stoploss: 3.0 * ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_kama_rsi_chop_1d_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio.
    """
    n = period
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if i >= n:
            price_change = np.abs(close[i] - close[i-n])
            noise = np.sum(np.abs(np.diff(close[i-n:i+1])))
            
            if noise > 0:
                er = price_change / noise
            else:
                er = 0.0
            
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = np.mean(close[max(0, i-n+1):i+1])
    
    return kama

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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 55/45 thresholds for clearer regime separation.
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend bias)
    kama_1d_21 = calculate_kama(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_10 = calculate_kama(close, period=10)
    kama_4h_21 = calculate_kama(close, period=21)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.30
    LONG_STRONG = 0.35
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_21[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP < 45 = trending market (use trend-following entries)
        # CHOP > 55 = range market (use mean-reversion entries)
        # CHOP 45-55 = transition (use either, lighter size)
        is_trending = chop_14[i] < 45.0
        is_choppy = chop_14[i] > 55.0
        
        # === 1D HTF TREND BIAS (direction filter, not hard block) ===
        # Bull bias: price above 1d KAMA (favor longs)
        # Bear bias: price below 1d KAMA (allow shorts)
        htF_bull = close[i] > kama_1d_21_aligned[i]
        htF_bear = close[i] < kama_1d_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        kama_bullish = kama_4h_10[i] > kama_4h_21[i]
        kama_bearish = kama_4h_10[i] < kama_4h_21[i]
        
        # KAMA slope (3-bar lookback)
        kama_slope_up = kama_4h_21[i] > kama_4h_21[i-3] if i >= 3 else False
        kama_slope_down = kama_4h_21[i] < kama_4h_4h_21[i-3] if i >= 3 else False
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_4h_21[i]
        price_below_kama = close[i] < kama_4h_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === RSI SIGNALS (widened ranges for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral_low = 38.0 < rsi_14[i] < 50.0
        rsi_neutral_high = 50.0 < rsi_14[i] < 62.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.998 if not np.isnan(donchian_upper[i]) else False
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.002 if not np.isnan(donchian_lower[i]) else False
        
        # === ENTRY LOGIC (REGIME-AWARE, SIMPLIFIED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Trend regime: follow 4h KAMA trend with HTF bias confirmation
        if is_trending:
            # KAMA bullish + RSI pullback + HTF bull bias
            if kama_bullish and rsi_neutral_low and (htF_bull or price_above_kama):
                new_signal = LONG_BASE
            
            # KAMA bullish crossover + RSI rising
            elif kama_bullish and kama_slope_up and rsi_rising and rsi_14[i] > 45.0:
                new_signal = LONG_BASE
            
            # Donchian breakout + trend confirmation
            elif donchian_breakout_up and kama_bullish and rsi_14[i] > 50.0:
                new_signal = LONG_STRONG
            
            # Strong oversold in bull regime
            elif rsi_strong_oversold and htF_bull:
                new_signal = LONG_BASE
        
        # Range regime: mean reversion at extremes
        if is_choppy:
            # RSI very oversold (mean revert long)
            if rsi_strong_oversold and price_below_kama:
                new_signal = LONG_BASE * 0.8
            
            # RSI oversold + near Donchian low
            elif rsi_oversold and close[i] < donchian_lower[i] * 1.02 if not np.isnan(donchian_lower[i]) else False:
                new_signal = LONG_BASE * 0.8
        
        # Transition regime (45-55): lighter entries
        if not is_trending and not is_choppy:
            if kama_bullish and rsi_neutral_low and htF_bull:
                new_signal = LONG_BASE * 0.7
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.7
        
        # SHORT ENTRIES (asymmetric - smaller size)
        if is_trending:
            # KAMA bearish + RSI pullback + HTF bear bias
            if kama_bearish and rsi_neutral_high and (htF_bear or price_below_kama):
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # KAMA bearish crossover + RSI falling
            elif kama_bearish and kama_slope_down and rsi_falling and rsi_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Donchian breakdown + trend confirmation
            elif donchian_breakout_down and kama_bearish and rsi_14[i] < 50.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # Strong overbought in bear regime
            elif rsi_strong_overbought and htF_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
        
        if is_choppy:
            # RSI very overbought (mean revert short)
            if rsi_strong_overbought and price_above_kama:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
            
            # RSI overbought + near Donchian high
            elif rsi_overbought and close[i] > donchian_upper[i] * 0.98 if not np.isnan(donchian_upper[i]) else False:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        if not is_trending and not is_choppy:
            if kama_bearish and rsi_neutral_high and htF_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7
            elif rsi_strong_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 20 bars (~80 hours = 3.3 days)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htF_bull and rsi_14[i] > 45.0 and kama_bullish:
                new_signal = LONG_BASE * 0.6
            elif htF_bear and rsi_14[i] < 55.0 and kama_bearish:
                new_signal = -SHORT_BASE * 0.6
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.6
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === KAMA REVERSAL EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and price_below_kama:
                kama_exit = True
            if position_side < 0 and kama_bullish and price_above_kama:
                kama_exit = True
        
        # === HTF REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htF_bear and price_below_kama:
                regime_reversal = True
            if position_side < 0 and htF_bull and price_above_kama:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or kama_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.32:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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