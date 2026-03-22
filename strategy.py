#!/usr/bin/env python3
"""
Experiment #043: 15m Fisher Transform + KAMA Trend + Choppiness Regime
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2022-2024).
Combined with KAMA (adaptive to volatility) to reduce whipsaws vs EMA/HMA.
Choppiness Index (CHOP) detects regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (trend follow).
This adapts strategy to market conditions instead of using one approach always.
Timeframe: 15m (REQUIRED), HTF: 1h via mtf_data helper for trend confirmation.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: CHOP regime filter switches between Fisher mean-reversion and KAMA trend-follow.
Simpler entry conditions to ensure >=10 trades per symbol (learned from 0-trade failures).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_kama_chop_regime_1h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for identifying turning points in bear/range markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        # Avoid division by zero
        if hh == ll:
            fisher[i] = fisher[i - 1] if i > period else 0.0
            continue
        
        # Normalize price to range -1 to +1
        value = (2 * (close[i] - ll) / (hh - ll)) - 1
        value = np.clip(value, -0.999, 0.999)  # Avoid log(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) - adapts to market volatility.
    Reduces whipsaws in ranging markets, follows trends in trending markets.
    Better than EMA for crypto's volatile nature.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (favor mean reversion)
    CHOP < 38.2 = trending market (favor trend following)
    Range 38.2-61.8 = transition zone
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh == ll:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        # CHOP formula
        chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

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
    """Calculate RSI with proper min_periods."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi[np.isnan(rsi)] = 50.0
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    kama_1h = calculate_kama(df_1h['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1h_aligned = align_htf_to_ltf(prices, df_1h, kama_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    kama_15m = calculate_kama(close, period=10)
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_HALF = 0.15
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    profit_target_hit = False
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_15m[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # 1h trend bias (HTF)
        bull_trend_1h = close[i] > kama_1h_aligned[i]
        bear_trend_1h = close[i] < kama_1h_aligned[i]
        
        # 15m trend
        bull_trend_15m = close[i] > kama_15m[i] and close[i] > ema_50[i]
        bear_trend_15m = close[i] < kama_15m[i] and close[i] < ema_50[i]
        
        # Choppiness regime
        range_regime = chop[i] > 55  # Ranging market
        trend_regime = chop[i] < 45  # Trending market
        # Middle zone 45-55 = transition, use either strategy
        
        # Fisher Transform signals
        fisher_long = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]  # Crossing up from oversold
        fisher_short = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]  # Crossing down from overbought
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # RSI confirmation (loose filter to ensure trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        new_signal = 0.0
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        
        # RANGE REGIME (CHOP > 55): Mean reversion with Fisher
        if range_regime:
            # Long: Fisher extreme oversold + RSI confirmation
            if fisher_extreme_long and rsi_oversold:
                new_signal = SIZE_BASE
            # Short: Fisher extreme overbought + RSI confirmation
            elif fisher_extreme_short and rsi_overbought:
                new_signal = -SIZE_BASE
            # Secondary: Fisher crossover + HTF trend alignment
            elif fisher_long and bull_trend_1h:
                new_signal = SIZE_BASE
            elif fisher_short and bear_trend_1h:
                new_signal = -SIZE_BASE
        
        # TREND REGIME (CHOP < 45): Trend following with KAMA
        elif trend_regime:
            # Long: Price above KAMA + HTF bullish + Fisher not overbought
            if bull_trend_15m and bull_trend_1h and fisher[i] < 1.0:
                new_signal = SIZE_BASE
            # Short: Price below KAMA + HTF bearish + Fisher not oversold
            elif bear_trend_15m and bear_trend_1h and fisher[i] > -1.0:
                new_signal = -SIZE_BASE
        
        # TRANSITION ZONE (45 <= CHOP <= 55): Mixed approach
        else:
            # Conservative entries only
            if fisher_extreme_long and bull_trend_1h:
                new_signal = SIZE_BASE
            elif fisher_extreme_short and bear_trend_1h:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            
            # Take profit at 2R, reduce to half position
            if not profit_target_hit and close[i] >= entry_price + 2.0 * 2.5 * atr[i]:
                profit_target_hit = True
                if new_signal == 0.0:
                    new_signal = SIZE_HALF  # Reduce to half
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            
            # Take profit at 2R, reduce to half position
            if not profit_target_hit and close[i] <= entry_price - 2.0 * 2.5 * atr[i]:
                profit_target_hit = True
                if new_signal == 0.0:
                    new_signal = -SIZE_HALF  # Reduce to half
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            profit_target_hit = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            profit_target_hit = False
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            profit_target_hit = False
        
        # Position reduced to half (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and abs(new_signal) < abs(prev_signal):
            profit_target_hit = True
        
        signals[i] = new_signal
    
    return signals