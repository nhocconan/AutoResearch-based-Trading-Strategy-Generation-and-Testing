#!/usr/bin/env python3
"""
Experiment #184: 4h KAMA Trend + 1d HMA Filter + RSI Pullback + CHOP Regime + ATR Stop

Hypothesis: 4h timeframe with KAMA (adaptive to volatility) captures trends better than 
static EMAs. Key innovation: Choppiness Index (CHOP) regime filter to distinguish 
trending vs ranging markets and apply different logic. 1d HMA provides stable HTF bias.
RSI pullback entries (not breakouts) reduce whipsaws. Conservative sizing (0.25) 
controls drawdown.

Why this might work on 4h:
- KAMA adapts smoothing based on market efficiency (fast in trends, slow in ranges)
- CHOP > 61.8 = range (use mean reversion), CHOP < 38.2 = trend (use trend following)
- 1d HMA filter prevents counter-trend trades
- RSI pullback (30-70) entries are more reliable than breakouts
- ATR 2.5x stoploss protects against reversals
- Discrete signal levels minimize fee churn

Learning from failures:
- #178 (4h Donchian): Sharpe=-0.989 - breakouts whipsaw in 4h noise
- #181 (15m CRSI): Sharpe=-5.141 - mean reversion alone fails
- #183 (1h vol spike): Sharpe=-3.477 - vol filters too restrictive
- Pure trend following fails in ranges, pure mean reversion fails in trends
- Need REGIME-ADAPTIVE logic (CHOP index)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_1d_hma_chop_regime_rsi_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Direction = abs(close - close[n-periods ago])
    # Noise = sum of abs(close[i] - close[i-1])
    
    for i in range(er_period, n):
        direction = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        
        if noise == 0:
            er = 1.0
        else:
            er = direction / noise
        
        # Smoothing constant
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period]
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_chop(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR(1), n) / (Highest High(n) - Lowest Low(n))) / LOG10(n)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR(1) = True Range for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        # Sum of ATR(1) over period
        atr_sum = np.sum(tr[i-period+1:i+1])
        
        # Highest High and Lowest Low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_hl = hh - ll
        
        if range_hl > 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 100  # Max choppy if no range
    
    # Fill initial values
    chop[:period] = chop[period] if period < n else 50
    
    return chop

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    chop = calculate_chop(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.15
    
    # Track position state for stoploss and take profit
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    tp_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend direction
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging market (mean reversion favorable)
        # CHOP < 38.2 = trending market (trend following favorable)
        # 38.2 - 61.8 = transition zone (use trend following with caution)
        is_choppy = chop[i] > 55  # Slightly lower threshold to catch more ranges
        is_trending = chop[i] < 45  # Slightly higher threshold to catch more trends
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 = sufficient trend strength (lower for 4h to ensure trades)
        trend_strength = adx[i] > 18
        
        # === KAMA TREND DIRECTION ===
        # Price above KAMA = bullish, below = bearish
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === EMA STRUCTURE ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # In uptrend: RSI pulling back to 40-50 zone = buy opportunity
        # In downtrend: RSI bouncing to 50-60 zone = sell opportunity
        rsi_pullback_long = 35 < rsi[i] < 55
        rsi_pullback_short = 45 < rsi[i] < 65
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        
        # --- TRENDING REGIME (CHOP < 45) ---
        if is_trending:
            # Long: 1d bullish + KAMA bullish + ADX strong + RSI not overbought
            if bull_trend_1d and kama_bullish and trend_strength:
                if rsi_pullback_long or (rsi_oversold and ema_bullish):
                    new_signal = SIZE_BASE
            
            # Short: 1d bearish + KAMA bearish + ADX strong + RSI not oversold
            if bear_trend_1d and kama_bearish and trend_strength:
                if rsi_pullback_short or (rsi_overbought and ema_bearish):
                    new_signal = -SIZE_BASE
        
        # --- CHOPPY/RANGING REGIME (CHOP > 55) ---
        elif is_choppy:
            # Mean reversion: buy oversold, sell overbought
            # Only trade with 1d trend bias for safety
            
            if bull_trend_1d and rsi_oversold:
                new_signal = SIZE_BASE
            
            if bear_trend_1d and rsi_overbought:
                new_signal = -SIZE_BASE
        
        # --- TRANSITION ZONE (45 <= CHOP <= 55) ---
        else:
            # Conservative: only trade strong signals with all confirmations
            if bull_trend_1d and kama_bullish and ema_bullish and rsi_oversold:
                new_signal = SIZE_BASE
            
            if bear_trend_1d and kama_bearish and ema_bearish and rsi_overbought:
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
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT LOGIC ===
        # Reduce position to half at 2R profit, let rest run with trailing stop
        if in_position and not tp_hit:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[i]:
                    new_signal = SIZE_HALF if new_signal > 0 else new_signal
                    tp_hit = True
            
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[i]:
                    new_signal = -SIZE_HALF if new_signal < 0 else new_signal
                    tp_hit = True
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                tp_hit = False
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                tp_hit = False
            elif abs(new_signal) < abs(position_side * SIZE_BASE):
                # Partial exit (take profit already hit)
                tp_hit = True
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                tp_hit = False
        
        signals[i] = new_signal
    
    return signals