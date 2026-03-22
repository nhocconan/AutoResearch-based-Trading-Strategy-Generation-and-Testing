#!/usr/bin/env python3
"""
Experiment #258: 1d KAMA Trend + RSI Pullback with 1w HMA Macro Bias

Hypothesis: Daily timeframe captures major swing moves (weeks to months) with high signal quality.
Using KAMA (Kaufman Adaptive Moving Average) which adapts to market volatility - faster in trends,
slower in ranges. Combined with RSI pullback entries and 1w HMA for macro directional bias.

Why this might work on 1d:
- KAMA adapts to volatility regime automatically (no need for separate regime filter)
- 1d has less noise than lower TFs - signals are more reliable
- 1w HMA provides macro bias without being too restrictive
- RSI(14) pullback to 40-60 range in trend = high win rate entries
- Fewer trades = less fee drag, each trade has more significance
- Conservative sizing (0.30) + 3*ATR stoploss controls drawdown on 2022 crash

Key differences from failed 1d strategies (#246, #252):
- KAMA instead of HMA (better volatility adaptation)
- Looser RSI entry range (35-65 instead of tight 42-48)
- 1w bias is soft (size modifier, not hard filter)
- Ensure minimum trade frequency with multiple entry paths

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 3.0 * ATR(14) trailing (wider for daily timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_pullback_1w_hma_atr_v1"
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
    KAMA adapts to market volatility - moves fast in trends, slow in ranges.
    
    Efficiency Ratio (ER) = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    Smoothing Constant (SC) = [ER * (fast - slow) + slow]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    rsi_14 = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_price_idx = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = macro trend bias (soft filter, size modifier)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND FILTER ===
        # Long-term trend via EMA200
        bull_trend_lt = close[i] > ema_200[i]
        bear_trend_lt = close[i] < ema_200[i]
        
        # Medium-term trend via KAMA crossover
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        kama_aligned_long = kama_fast[i] > kama_slow[i] and kama_fast[i] > kama_fast[i-5]
        kama_aligned_short = kama_fast[i] < kama_slow[i] and kama_fast[i] < kama_fast[i-5]
        
        # === TREND STRENGTH ===
        trend_strong = adx[i] > 20  # ADX > 20 = trending market
        trend_weak = adx[i] < 20    # ADX < 20 = ranging market
        
        # DI crossover for direction
        di_cross_long = plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]
        di_cross_short = plus_di[i] < minus_di[i] and plus_di[i-1] >= minus_di[i-1]
        
        # === RSI PULLBACK ENTRIES ===
        # Long: RSI pulled back to 35-55 in uptrend
        rsi_pullback_long = 35 <= rsi_14[i] <= 55 and close[i] > ema_50[i]
        # Short: RSI rallied to 45-65 in downtrend
        rsi_pullback_short = 45 <= rsi_14[i] <= 65 and close[i] < ema_50[i]
        
        # RSI extreme reversal (counter-trend, only in weak trend)
        rsi_oversold = rsi_14[i] < 30 and trend_weak
        rsi_overbought = rsi_14[i] > 70 and trend_weak
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        long_score = 0.0
        short_score = 0.0
        
        # KAMA crossover (strong signal)
        if kama_cross_long:
            long_score += 2.0
        if kama_cross_short:
            short_score += 2.0
        
        # KAMA alignment (medium signal)
        if kama_aligned_long:
            long_score += 1.0
        if kama_aligned_short:
            short_score += 1.0
        
        # RSI pullback (medium signal)
        if rsi_pullback_long:
            long_score += 1.5
        if rsi_pullback_short:
            short_score += 1.5
        
        # DI crossover (confirmation)
        if di_cross_long:
            long_score += 1.0
        if di_cross_short:
            short_score += 1.0
        
        # RSI extreme (counter-trend, lower weight)
        if rsi_oversold:
            long_score += 0.5
        if rsi_overbought:
            short_score += 0.5
        
        # === APPLY TREND FILTERS ===
        # Long entries need bullish bias (either LT trend or 1w HMA)
        if long_score >= 2.5:
            if bull_trend_lt or bull_trend_1w:
                new_signal = SIZE_BASE
            elif trend_weak:  # Allow in range market
                new_signal = SIZE_BASE * 0.7
        
        # Short entries need bearish bias
        if short_score >= 2.5:
            if bear_trend_lt or bear_trend_1w:
                new_signal = -SIZE_BASE
            elif trend_weak:  # Allow in range market
                new_signal = -SIZE_BASE * 0.7
        
        # Boost size if 1w agrees (stronger conviction)
        if new_signal > 0 and bull_trend_1w and bull_trend_lt:
            new_signal = min(new_signal + 0.05, 0.35)
        if new_signal < 0 and bear_trend_1w and bear_trend_lt:
            new_signal = max(new_signal - 0.05, -0.35)
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.0 * ATR below highest close
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.0 * ATR above lowest close
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2.5R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.5 * atr[entry_price_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.5 * atr[entry_price_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
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