#!/usr/bin/env python3
"""
Experiment #012: 12h Dual-Regime Strategy with KAMA/ADX/Bollinger

Hypothesis: Single-regime strategies fail because crypto alternates between
trending and ranging markets. This strategy detects regime and switches logic:

1. REGIME DETECTION (ADX + Choppiness):
   - ADX > 25 = Trending regime → use KAMA trend following
   - ADX < 20 = Ranging regime → use Bollinger mean reversion
   - Hysteresis (25/20) prevents rapid regime switching

2. TREND REGIME (ADX > 25):
   - Long: KAMA(10) > KAMA(30) + price > KAMA(10) + RSI(14) > 45
   - Short: KAMA(10) < KAMA(30) + price < KAMA(10) + RSI(14) < 55
   - 1d HMA confirms direction (only trade with HTF trend)

3. RANGE REGIME (ADX < 20):
   - Long: price < BB_lower + RSI(14) < 35 + 1w HMA bullish
   - Short: price > BB_upper + RSI(14) > 65 + 1w HMA bearish
   - Exit when RSI crosses 50 (mean reached)

4. HTF FILTERS:
   - 1d HMA(21): Intermediate trend bias
   - 1w HMA(21): Major trend bias (only for range regime entries)

5. RISK MANAGEMENT:
   - Position size: 0.28 (discrete, conservative)
   - Stoploss: 2.5 * ATR(14) trailing
   - Timeframe: 12h (targets 25-45 trades/year)

Why this should work:
- Adapts to market conditions (major failure mode of previous strategies)
- 12h TF has proven success (#002 had +13% return despite negative Sharpe)
- Hysteresis prevents whipsaw between regimes
- HTF filters reduce counter-trend trades (2022 crash killer)
- Conservative sizing protects against extreme drawdowns

Timeframe: 12h (REQUIRED for Experiment #012)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_kama_adx_bb_1d_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change = absolute price change over er_period
    change = np.abs(close_s.diff(er_period))
    
    # Volatility = sum of absolute single-period changes
    volatility = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (ER) = change / volatility
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1] if i > 0 else close[i]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = ranging
    """
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
    
    # Smoothed values (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    # Bandwidth for regime confirmation
    bandwidth = (upper - lower) / sma
    
    return upper.values, lower.values, sma.values, bandwidth.values

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
    
    return chop.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_30 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    # Regime state tracking (hysteresis)
    prev_regime = None  # 'trend' or 'range'
    regime_lock_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (with hysteresis) ===
        adx_value = adx_14[i]
        chop_value = chop_14[i]
        
        # Determine current regime
        if adx_value > 25 or chop_value < 40:
            current_regime = 'trend'
        elif adx_value < 20 or chop_value > 55:
            current_regime = 'range'
        else:
            # Neutral zone - keep previous regime (hysteresis)
            current_regime = prev_regime if prev_regime is not None else 'range'
        
        # Lock regime for minimum 20 bars to prevent flipping
        if current_regime != prev_regime and (i - regime_lock_bar) > 20:
            prev_regime = current_regime
            regime_lock_bar = i
        
        # === HTF TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral_low = rsi_14[i] < 45
        rsi_neutral_high = rsi_14[i] > 55
        
        # === BOLLINGER BAND CONDITIONS ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        bb_near_lower = close[i] < bb_mid[i] and close[i] > bb_lower[i] * 0.99
        bb_near_upper = close[i] > bb_mid[i] and close[i] < bb_upper[i] * 1.01
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        if prev_regime == 'trend':
            # === TREND REGIME: Follow KAMA direction ===
            
            # LONG: KAMA bullish + price above KAMA + RSI confirmation + HTF bias
            long_trend_score = 0
            
            if kama_bullish:
                long_trend_score += 2.0
            if close[i] > kama_10[i]:
                long_trend_score += 1.0
            if rsi_neutral_low or rsi_14[i] > 50:
                long_trend_score += 1.0
            if daily_bullish:
                long_trend_score += 1.0
            
            # Enter long if score >= 4.0
            if long_trend_score >= 4.0:
                new_signal = BASE_SIZE
            
            # SHORT: KAMA bearish + price below KAMA + RSI confirmation + HTF bias
            short_trend_score = 0
            
            if kama_bearish:
                short_trend_score += 2.0
            if close[i] < kama_10[i]:
                short_trend_score += 1.0
            if rsi_neutral_high or rsi_14[i] < 50:
                short_trend_score += 1.0
            if daily_bearish:
                short_trend_score += 1.0
            
            # Enter short if score >= 4.0
            if short_trend_score >= 4.0:
                new_signal = -BASE_SIZE
        
        else:
            # === RANGE REGIME: Mean reversion at BB extremes ===
            
            # LONG: Price at BB lower + RSI oversold + weekly bias (optional)
            long_range_score = 0
            
            if bb_oversold or bb_near_lower:
                long_range_score += 2.0
            if rsi_oversold:
                long_range_score += 2.0
            elif rsi_14[i] < 40:
                long_range_score += 1.5
            if weekly_bullish or daily_bullish:
                long_range_score += 1.0
            
            # Enter long if score >= 4.0
            if long_range_score >= 4.0:
                new_signal = BASE_SIZE
            
            # SHORT: Price at BB upper + RSI overbought + weekly bias (optional)
            short_range_score = 0
            
            if bb_overbought or bb_near_upper:
                short_range_score += 2.0
            if rsi_overbought:
                short_range_score += 2.0
            elif rsi_14[i] > 60:
                short_range_score += 1.5
            if weekly_bearish or daily_bearish:
                short_range_score += 1.0
            
            # Enter short if score >= 4.0
            if short_range_score >= 4.0:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~50 days on 12h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if prev_regime == 'trend':
                if kama_bullish and daily_bullish and rsi_14[i] > 45:
                    new_signal = REDUCED_SIZE
                elif kama_bearish and daily_bearish and rsi_14[i] < 55:
                    new_signal = -REDUCED_SIZE
            else:
                if bb_oversold and rsi_14[i] < 40:
                    new_signal = REDUCED_SIZE
                elif bb_overbought and rsi_14[i] > 60:
                    new_signal = -REDUCED_SIZE
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and prev_regime == 'range' and bb_overbought:
                regime_exit = True  # Long in range regime at BB upper
            if position_side < 0 and prev_regime == 'range' and bb_oversold:
                regime_exit = True  # Short in range regime at BB lower
        
        # === RSI MEAN REVERSION EXIT (range regime) ===
        rsi_exit = False
        if in_position and position_side != 0 and prev_regime == 'range':
            # Exit when RSI crosses back to neutral (mean reached)
            if position_side > 0 and rsi_14[i] > 55:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 45:
                rsi_exit = True
        
        # === KAMA REVERSAL EXIT (trend regime) ===
        kama_exit = False
        if in_position and position_side != 0 and prev_regime == 'trend':
            # Exit when KAMA crossover reverses
            if position_side > 0 and kama_bearish:
                kama_exit = True
            if position_side < 0 and kama_bullish:
                kama_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_exit or rsi_exit or kama_exit:
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