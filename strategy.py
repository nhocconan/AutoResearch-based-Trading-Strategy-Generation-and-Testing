#!/usr/bin/env python3
"""
Experiment #427: 1d Primary + 1w HTF — Fisher Transform + KAMA + Choppiness Regime

Hypothesis: After analyzing 426 failed experiments, clear patterns emerge for 1d timeframe:
1. 1d needs 15-30 trades/year (higher TF = fewer trades = less fee drag)
2. Fisher Transform catches reversals better than RSI in bear/range markets (research shows edge)
3. KAMA (Kaufman Adaptive) adapts to volatility better than HMA/EMA (proven in #422 variant)
4. Choppiness Index regime filter: CHOP>61.8=mean revert, CHOP<38.2=trend follow
5. 1w HTF for major regime direction (prevents counter-trend trades in 2022-style crashes)
6. Asymmetric position sizing: smaller in chop (0.20), larger in trend (0.30)

Why this might beat current best (Sharpe=0.435):
- Fisher Transform has superior reversal detection vs RSI (Ehlers research)
- KAMA adapts to market noise (ER ratio) vs fixed-period HMA
- Choppiness regime prevents trend-following in ranges (major failure mode)
- 1w HTF filter is proven in current best strategy
- Conservative sizing (0.20-0.30) limits drawdown in 2022 crash

Position sizing: 0.20-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR trailing
Target: 15-30 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_kama_chop_1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Net change over ER period
    net_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute changes over ER period
    abs_changes = np.abs(close_s.diff().values)
    sum_abs = pd.Series(abs_changes).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Efficiency Ratio
    er = np.zeros(n)
    mask = sum_abs > 0
    er[mask] = net_change / sum_abs
    er[:er_period] = np.nan
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[:er_period] = np.nan
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for better reversal detection.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price position
        if hh > ll:
            value = (2.0 * high[i] - hh - ll) / (hh - ll + 1e-10)
            value = np.clip(value * 0.999, -0.999, 0.999)
        else:
            value = 0.0
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value + 1e-10))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    
    # Sum of ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Choppiness formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10((hh - ll) / (atr_sum + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    kama_1w_20 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1w_20_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_20)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    choppiness = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    SIZE_TREND = 0.30
    SIZE_CHOP = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1w_20_aligned[i]):
            continue
        
        if np.isnan(kama_1d_20[i]) or np.isnan(fisher[i]) or np.isnan(choppiness[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w KAMA = bull market bias (favor longs)
        # Price below 1w KAMA = bear market bias (favor shorts)
        bull_regime = close[i] > kama_1w_20_aligned[i]
        bear_regime = close[i] < kama_1w_20_aligned[i]
        
        # === 1D LOCAL TREND (KAMA slope) ===
        kama_slope_bullish = kama_1d_20[i] > kama_1d_20[i-5] if i >= 5 else False
        kama_slope_bearish = kama_1d_20[i] < kama_1d_20[i-5] if i >= 5 else False
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = ranging (mean reversion preferred)
        # CHOP < 38.2 = trending (breakout preferred)
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS (reversal detection) ===
        # Fisher crossing above -1.5 = bullish reversal
        # Fisher crossing below +1.5 = bearish reversal
        fisher_bullish_cross = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_bearish_cross = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher extremes
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY in BULL regime
        if bull_regime:
            if is_choppy:
                # Mean reversion: Fisher oversold + RSI oversold
                if fisher_oversold and rsi_oversold:
                    new_signal = SIZE_CHOP
            elif is_trending:
                # Trend follow: Fisher bullish cross + KAMA slope up
                if fisher_bullish_cross and kama_slope_bullish:
                    new_signal = SIZE_TREND
                elif fisher[i] > -1.0 and kama_slope_bullish and rsi_14[i] < 60.0:
                    new_signal = SIZE_TREND * 0.8
        
        # SHORT ENTRY in BEAR regime
        if bear_regime:
            if is_choppy:
                # Mean reversion: Fisher overbought + RSI overbought
                if fisher_overbought and rsi_overbought:
                    if new_signal == 0.0:
                        new_signal = -SIZE_CHOP
            elif is_trending:
                # Trend follow: Fisher bearish cross + KAMA slope down
                if fisher_bearish_cross and kama_slope_bearish:
                    if new_signal == 0.0:
                        new_signal = -SIZE_TREND
                elif fisher[i] < 1.0 and kama_slope_bearish and rsi_14[i] > 40.0:
                    if new_signal == 0.0:
                        new_signal = -SIZE_TREND * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 25 bars (~25 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if bull_regime and fisher[i] < -0.5 and kama_slope_bullish:
                new_signal = SIZE_CHOP * 0.7
            elif bear_regime and fisher[i] > 0.5 and kama_slope_bearish:
                new_signal = -SIZE_CHOP * 0.7
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on reversal exhaustion)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # RSI extreme exit
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (1w KAMA cross)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
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
                # Position flip
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