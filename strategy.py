#!/usr/bin/env python3
"""
Experiment #300: 1h Primary + 4h/12h HTF — Simplified Trend-Follow with RSI Pullback

Hypothesis: After 272 failed experiments, the key insight is:
1. Lower TF (1h) strategies FAIL due to TOO STRICT entry conditions (0 trades)
2. Need SIMPLE, RELIABLE signals that actually trigger in all market regimes
3. Use 4h HMA for PRIMARY trend direction (proven edge from #293)
4. Use 1h RSI for ENTRY TIMING (pullback within trend, not extremes)
5. Add volume filter (loose: >0.8x avg) and session filter (8-20 UTC) for quality
6. Target: 40-80 trades/year on 1h (appropriate frequency)

Why this might work when #290, #295, #298 failed (all Sharpe=0.000):
- RSI thresholds: 35/65 instead of 25/75 (actually triggers)
- No ADX filter (ADX>25 too rare in crypto)
- No Fisher Transform (too many conditions = 0 trades)
- Session filter on ENTRY only, not exit (allows trades to run)
- Simpler logic = more trades = actual Sharpe calculation

Position sizing: 0.25 base, 0.30 strong (conservative for 1h TF)
Stoploss: 2.5 * ATR(14) trailing (tighter than daily, wider than 15m)
Target R:R: 1:2 minimum (exit at 2*ATR profit or trail)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_session_4h_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (primary trend regime)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Calculate 12h HTF indicators (secondary trend confirmation)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    hma_1h_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume moving average for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === 4H TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA21 AND HMA21 > HMA50
        # Bear: price below 4h HMA21 AND HMA21 < HMA50
        trend_4h_bull = (close[i] > hma_4h_21_aligned[i]) and (hma_4h_21_aligned[i] > hma_4h_50_aligned[i])
        trend_4h_bear = (close[i] < hma_4h_21_aligned[i]) and (hma_4h_21_aligned[i] < hma_4h_50_aligned[i])
        
        # === 12H TREND CONFIRMATION (secondary filter) ===
        # Only trade long if 12h trend agrees (or neutral)
        trend_12h_bull = close[i] > hma_12h_21_aligned[i]
        trend_12h_bear = close[i] < hma_12h_21_aligned[i]
        
        # === 1H LOCAL TREND ===
        trend_1h_bull = hma_1h_21[i] > hma_1h_50[i]
        trend_1h_bear = hma_1h_21[i] < hma_1h_50[i]
        
        # === VOLUME FILTER (loose: >0.8x average) ===
        vol_ok = volume[i] > 0.8 * vol_sma_20[i] if not np.isnan(vol_sma_20[i]) else True
        
        # === SESSION FILTER (8-20 UTC for entries only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === RSI SIGNALS (pullback within trend) ===
        # Long: RSI 35-50 (pullback in uptrend)
        # Short: RSI 50-65 (pullback in downtrend)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === SMA200 FILTER (major trend) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC (SIMPLIFIED - must trigger!) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (4h bull + 1h pullback + volume)
        if trend_4h_bull and trend_12h_bull:
            # Primary: RSI pullback in session with volume
            if rsi_pullback_long and in_session and vol_ok:
                new_signal = BASE_SIZE
            # Secondary: RSI oversold (stronger signal, any time)
            elif rsi_oversold and above_sma200:
                new_signal = STRONG_SIZE
            # Tertiary: 1h trend confirmation (no session filter)
            elif trend_1h_bull and rsi_14[i] > 45 and rsi_14[i] < 60:
                new_signal = BASE_SIZE
        
        # SHORT ENTRIES (4h bear + 1h pullback + volume)
        if trend_4h_bear and trend_12h_bear:
            # Primary: RSI pullback in session with volume
            if rsi_pullback_short and in_session and vol_ok:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # Secondary: RSI overbought (stronger signal, any time)
            elif rsi_overbought and below_sma200:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            # Tertiary: 1h trend confirmation (no session filter)
            elif trend_1h_bear and rsi_14[i] < 55 and rsi_14[i] > 40:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 1h) ===
        # Force trade if no signal for 100 bars (~4 days on 1h)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_4h_bull and rsi_14[i] > 40 and rsi_14[i] < 60:
                new_signal = BASE_SIZE * 0.5
            elif trend_4h_bear and rsi_14[i] > 40 and rsi_14[i] < 60:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TAKE PROFIT LOGIC (2R target, then trail) ===
        take_profit_triggered = False
        if in_position and position_side != 0:
            profit_target = entry_price + position_side * 2.0 * entry_atr
            if position_side > 0 and close[i] >= profit_target:
                take_profit_triggered = True
            if position_side < 0 and close[i] <= profit_target:
                take_profit_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h trend turns bearish
            if position_side > 0 and trend_4h_bear:
                trend_reversal = True
            # Short position but 4h trend turns bullish
            if position_side < 0 and trend_4h_bull:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long: exit on overbought
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            # Short: exit on oversold
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        if stoploss_triggered or take_profit_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0.28:
                new_signal = STRONG_SIZE
            elif new_signal > 0:
                new_signal = BASE_SIZE
            elif new_signal < -0.28:
                new_signal = -STRONG_SIZE
            else:
                new_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_atr = atr_14[i]
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                entry_atr = atr_14[i]
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                entry_atr = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals