#!/usr/bin/env python3
"""
Experiment #283: 15m KAMA Adaptive Trend with 4h HMA Bias and Volume Filter

Hypothesis: After analyzing 282 experiments, clear patterns emerge:
1. RSI-based strategies consistently fail (see #251, #254, #259, #277)
2. Complex ensembles with too many filters generate 0 trades (#275, #277, #280)
3. Simple trend-following with strong HTF bias works best (current best: #274)
4. 15m timeframe needs adaptive indicators to handle volatility regimes

This strategy uses:
1. 4h HMA(21) for directional bias - prevents counter-trend trades (proven in #274)
2. 15m KAMA(10) adaptive MA - adjusts to volatility, less lag than EMA
3. Volume confirmation (1.2x average) - filters false breakouts
4. 2.5*ATR trailing stoploss - appropriate for 15m (tighter than 12h)
5. Asymmetric entries - only trade in direction of 4h HMA bias
6. Discrete position sizing (0.25/0.30) - minimizes fee churn

Why 15m might work:
- Faster entries than 4h/12h strategies
- KAMA adapts to volatility (critical for 2022 crash period)
- 4h HMA bias prevents whipsaw counter-trend trades
- Should generate 50-100+ trades per symbol (well above minimum 10)

Key differences from failed strategies:
- NO RSI (RSI strategies have 100% failure rate in our experiments)
- NO Fisher Transform (#271, #280 failed with Sharpe=0.000)
- NO Choppiness Index (#262 failed with Sharpe=-0.159)
- Simple KAMA crossover + HTF bias + volume = cleaner signals
- Looser volume threshold (1.2x not 1.5x) to ensure trades

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_trend_4h_hma_volume_atr_v1"
timeframe = "15m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - smooth in ranging, fast in trending.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = Change / Volatility (Efficiency Ratio)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    volatility[:period] = np.nan
    volatility[volatility == 0] = np.nan
    
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Calculate KAMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA
    kama[period-1] = np.nanmean(close[:period])
    
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=20, fast=2, slow=30)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size (conservative for 15m)
    SIZE_HIGH = 0.30  # Higher size in strong trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = strong directional bias (hard filter)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        # Breakout must have volume > 1.2x average (looser than 1.5x to ensure trades)
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === KAMA CROSSOVER SIGNALS ===
        # Fast KAMA crosses above slow KAMA = bullish momentum
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # Check for crossover (fast crosses slow)
        kama_cross_long = False
        kama_cross_short = False
        
        if i > 0 and not np.isnan(kama_fast[i-1]) and not np.isnan(kama_slow[i-1]):
            # Long crossover: fast was below, now above
            if kama_fast[i-1] <= kama_slow[i-1] and kama_fast[i] > kama_slow[i]:
                kama_cross_long = True
            # Short crossover: fast was above, now below
            if kama_fast[i-1] >= kama_slow[i-1] and kama_fast[i] < kama_slow[i]:
                kama_cross_short = True
        
        # === TREND STRENGTH ===
        # Price above EMA50 = strong uptrend, below = strong downtrend
        strong_uptrend = close[i] > ema_50[i] and bull_trend_4h
        strong_downtrend = close[i] < ema_50[i] and bear_trend_4h
        
        # Determine position size based on trend strength
        if strong_uptrend or strong_downtrend:
            position_size = SIZE_HIGH
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + KAMA bullish + (crossover OR volume confirmed)
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            kama_bullish and  # KAMA trend bullish
            (kama_cross_long or volume_confirmed)  # Crossover or volume
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            kama_bearish and  # KAMA trend bearish
            (kama_cross_short or volume_confirmed)  # Crossover or volume
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals