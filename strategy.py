#!/usr/bin/env python3
"""
Experiment #186: 1d KAMA Trend + Choppiness Regime + RSI Entry + ATR Stop

Hypothesis: Daily timeframe captures major trend moves while filtering noise.
KAMA (Kaufman Adaptive MA) adapts to market efficiency - fast in trends, slow in ranges.
Choppiness Index detects regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend.
RSI extremes provide entry timing within the regime context.
ATR trailing stop protects against reversals on large daily bars.

Why 1d might work now:
- KAMA outperformed simple EMA in prior experiments (#173 Sharpe=0.192)
- Choppiness filter prevents trend strategies in ranges (major failure mode)
- RSI 35/65 thresholds (not extreme 20/80) ensure sufficient trade count on 1d
- 1d bars = ~1460 for 4 years, need ~15-25 trades minimum
- Conservative sizing (0.28) controls drawdown on large daily moves

Learning from failures:
- #174 (1d KAMA 1w HMA): Sharpe=-0.004 - needed regime filter
- #180 (1d KAMA 1w Donchian): Sharpe=-0.291 - Donchian too slow on 1d
- #185 (12h Chop Regime): Sharpe=-0.299 - 12h still too noisy
- Pure trend following fails in 2022 crash without regime detection
- Mean reversion alone fails on crypto trends

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_regime_rsi_atr_v1"
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
    KAMA adapts smoothing based on market efficiency ratio.
    Fast in trends (high ER), slow in ranges (low ER).
    """
    n = len(close)
    close = np.asarray(close, dtype=float)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period].mean()
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    Based on ATR and total price range over period.
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        price_range = hh - ll
        
        if price_range > 0:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 100
    
    # Fill initial values
    chop[:period] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    kama_1w = calculate_kama(df_1w['close'].values, er_period=10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama_1d = calculate_kama(close, er_period=10)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(kama_1d[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w KAMA = higher timeframe trend bias
        bull_trend_1w = close[i] > kama_1w_aligned[i]
        bear_trend_1w = close[i] < kama_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range market (mean revert)
        # CHOP < 38.2 = trending market (trend follow)
        # 38.2 - 61.8 = neutral (use trend bias)
        is_range = chop[i] > 55.0  # Slightly lower threshold for more trades
        is_trend = chop[i] < 45.0  # Slightly higher threshold for more trades
        
        # === 1d KAMA TREND ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI ENTRY SIGNALS ===
        # In range: buy low RSI, sell high RSI
        # In trend: buy RSI pullback, sell RSI rally
        rsi_oversold = rsi[i] < 45  # Less extreme for more trades on 1d
        rsi_overbought = rsi[i] > 55  # Less extreme for more trades on 1d
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # LONG entries:
        if is_range:
            # Range regime: mean reversion
            # Long when RSI oversold + price above weekly KAMA (bullish bias)
            if rsi_oversold and bull_trend_1w:
                new_signal = SIZE_BASE
        elif is_trend:
            # Trend regime: trend following
            # Long when KAMA bullish + EMA bullish + RSI not overbought
            if kama_bullish and ema_bullish and rsi[i] < 65:
                new_signal = SIZE_BASE
        else:
            # Neutral regime: use trend bias + RSI pullback
            if bull_trend_1w and kama_bullish and rsi_oversold:
                new_signal = SIZE_BASE
        
        # SHORT entries:
        if is_range:
            # Range regime: mean reversion
            # Short when RSI overbought + price below weekly KAMA (bearish bias)
            if rsi_overbought and bear_trend_1w:
                new_signal = -SIZE_BASE
        elif is_trend:
            # Trend regime: trend following
            # Short when KAMA bearish + EMA bearish + RSI not oversold
            if kama_bearish and ema_bearish and rsi[i] > 35:
                new_signal = -SIZE_BASE
        else:
            # Neutral regime: use trend bias + RSI rally
            if bear_trend_1w and kama_bearish and rsi_overbought:
                new_signal = -SIZE_BASE
        
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
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif int(np.sign(new_signal)) != position_side:
                # Reversing position
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
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