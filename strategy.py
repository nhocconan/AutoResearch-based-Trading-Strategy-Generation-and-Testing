#!/usr/bin/env python3
"""
Experiment #199: 15m KAMA Trend + 4h HMA Bias + RSI Pullback + ATR Stop

Hypothesis: 15m timeframe with KAMA (adaptive MA) captures trends while filtering noise
better than EMA. 4h HMA provides stable higher-timeframe bias. RSI(7) pullback entries
within the trend direction capture better risk/reward than breakouts. ATR trailing stop
protects against reversals. This combines adaptive trend following with mean-reversion
entries within the trend - a hybrid approach that should work in both trending and
ranging markets.

Why this might work:
- KAMA adapts to volatility (fast in trends, slow in ranges) - proven in #198
- 4h HMA filter prevents counter-trend trades (multi-timeframe edge)
- RSI(7) pullback = enter on dips in uptrend, rallies in downtrend (better R/R)
- ADX > 15 ensures we're not trading dead chop
- ATR 2.5x stop protects capital in 2022-style crashes
- Conservative sizing (0.30) controls drawdown

Learning from failures:
- #193 (15m pullback): Sharpe=-3.489 - likely too many filters or wrong entry logic
- #187 (15m Supertrend): Sharpe=-1.239 - Supertrend whipsaws badly on 15m
- #198 (1d KAMA): Sharpe=0.028 - worked but 1d too slow, 15m should capture more moves
- Mean reversion alone fails (194: -2.588), need trend filter
- Too many filters = 0 trades, keep conditions flexible

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_4h_hma_rsi_pullback_atr_v1"
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
    """Calculate Kaufman's Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i-period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[period-1] = close[period-1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:period-1] = close[:period-1]
    
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama_21 = calculate_kama(close, 10, 2, 30)
    kama_50 = calculate_kama(close, 10, 2, 30)
    # For KAMA 50, recalculate with longer effective period
    kama_50 = calculate_kama(close, 20, 2, 30)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss and take profit
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    profit_target_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama_21[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = some trend (lower threshold for 15m to ensure trades)
        trend_strength = adx[i] > 15
        
        # === KAMA TREND STRUCTURE ===
        # KAMA21 > KAMA50 = bullish trend structure
        # KAMA21 < KAMA50 = bearish trend structure
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI(7) dips to 35-45 in uptrend (buy the dip)
        # Short: RSI(7) rallies to 55-65 in downtrend (sell the rip)
        rsi_pullback_long = 30 < rsi_7[i] < 50
        rsi_pullback_short = 50 < rsi_7[i] < 70
        
        # === PRICE VS KAMA ===
        # Long: price pulls back to/near KAMA21 but stays above
        # Short: price rallies to/near KAMA21 but stays below
        price_near_kama_long = close[i] > kama_21[i] * 0.995
        price_near_kama_short = close[i] < kama_21[i] * 1.005
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + ADX trending + KAMA bullish + RSI pullback + price near KAMA
        # Using OR logic to ensure enough trades
        if bull_trend_4h and trend_strength:
            if kama_bullish and rsi_pullback_long and price_near_kama_long:
                new_signal = SIZE_BASE
            elif kama_bullish and rsi_7[i] < 40:
                # Deep oversold in uptrend = strong buy signal
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + ADX trending + KAMA bearish + RSI pullback + price near KAMA
        if bear_trend_4h and trend_strength:
            if kama_bearish and rsi_pullback_short and price_near_kama_short:
                new_signal = -SIZE_BASE
            elif kama_bearish and rsi_7[i] > 60:
                # Deep overbought in downtrend = strong sell signal
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and new_signal != 0.0:
            # Keep existing position signal if not stopped out
            new_signal = signals[i-1] if signals[i-1] != 0.0 else new_signal
        
        if in_position:
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
        
        # === TAKE PROFIT - Reduce to half at 2R ===
        if in_position and not profit_target_hit:
            if position_side > 0:
                if close[i] >= entry_price + 2.0 * 2.5 * atr[i]:
                    new_signal = SIZE_HALF
                    profit_target_hit = True
            if position_side < 0:
                if close[i] <= entry_price - 2.0 * 2.5 * atr[i]:
                    new_signal = -SIZE_HALF
                    profit_target_hit = True
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif new_signal != signals[i-1] and profit_target_hit:
                # Already hit TP, maintain half position
                pass
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                profit_target_hit = False
        
        signals[i] = new_signal
    
    return signals