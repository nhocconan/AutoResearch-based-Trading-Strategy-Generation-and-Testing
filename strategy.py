#!/usr/bin/env python3
"""
Experiment #030: 1h RSI Pullback + 4h/12h HMA Trend + Volume/Session Filter

Hypothesis: 1h primary with 4h/12h HTF trend filter will capture pullback entries
within established trends. Key insight from failures:
- Connors RSI was too complex and failed (exp#028 Sharpe=-1.817)
- Simple RSI(14) pullback within HTF trend works better
- Need 3+ confluence but NOT over-filtered (learned from 0-trade failures)
- Session filter (8-20 UTC) ensures liquidity during entry
- Volume confirmation prevents fake breakouts

Design:
1. 4h HMA(21) slope for primary trend direction (call ONCE via mtf_data)
2. 12h HMA(21) for regime confirmation (bull/bear)
3. 1h RSI(14) pullback: long when RSI 35-50 in uptrend, short when 50-65 in downtrend
4. Volume > 0.8x 20-bar average (not too strict)
5. Session filter: only enter 8-20 UTC (liquidity hours)
6. ATR(14) stoploss at 2.5x
7. Discrete sizing: 0.20-0.30 (conservative for 1h TF)

Why this should work:
- RSI pullback entries have better win rate than breakouts in ranging markets
- 4h/12h HTF filter prevents counter-trend trades (major 2022 failure mode)
- Session filter reduces noise during low-liquidity hours
- Simple RSI thresholds (35-50, 50-65) ensure trades actually trigger
- Targets 30-60 trades/year on 1h (optimal for fee efficiency)

Timeframe: 1h (REQUIRED)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_12h_hma_session_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate HTF HMA trends
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to 1h timeframe (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Calculate HMA slope for 4h trend direction
    hma_4h_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-5]):
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-5]) / hma_4h_aligned[i-5]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 4H HTF TREND BIAS ===
        # Price above 4h HMA + positive slope = bullish
        # Price below 4h HMA + negative slope = bearish
        htf_bullish = close[i] > hma_4h_aligned[i] and hma_4h_slope[i] > 0.001
        htf_bearish = close[i] < hma_4h_aligned[i] and hma_4h_slope[i] < -0.001
        
        # === 12H REGIME CONFIRMATION ===
        # 12h HMA confirms major trend direction
        regime_bullish = close[i] > hma_12h_aligned[i]
        regime_bearish = close[i] < hma_12h_aligned[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI 35-50 (pullback in uptrend)
        # Short: RSI 50-65 (pullback in downtrend)
        rsi_pullback_long = 35 <= rsi_14[i] <= 50
        rsi_pullback_short = 50 <= rsi_14[i] <= 65
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.8x 20-bar average (not too strict)
        volume_ok = volume[i] > 0.8 * vol_sma_20[i] if vol_sma_20[i] > 0 else True
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only enter during high-liquidity hours
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        session_ok = 8 <= hour_utc <= 20
        
        # === POSITION SIZING BASED ON CONFLUENCE ===
        # Strong: 4h + 12h agree + volume + session
        # Base: 4h trend + RSI pullback
        # Weak: 4h trend only
        long_confluence = htf_bullish and regime_bullish and volume_ok and session_ok
        short_confluence = htf_bearish and regime_bearish and volume_ok and session_ok
        
        if long_confluence:
            current_size = STRONG_SIZE
        elif htf_bullish:
            current_size = BASE_SIZE
        elif short_confluence:
            current_size = STRONG_SIZE
        elif htf_bearish:
            current_size = BASE_SIZE
        else:
            current_size = WEAK_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + RSI pullback (35-50)
        if htf_bullish and rsi_pullback_long:
            # Add session/volume for full size, allow without for reduced size
            if session_ok and volume_ok:
                new_signal = current_size
            elif bars_since_last_trade > 24:  # Allow after 24 bars (~1 day)
                new_signal = current_size * 0.7
        
        # SHORT ENTRY: 4h bearish + RSI pullback (50-65)
        elif htf_bearish and rsi_pullback_short:
            if session_ok and volume_ok:
                new_signal = -current_size
            elif bars_since_last_trade > 24:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 48 bars (~2 days), allow weaker entry
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if htf_bullish and 30 <= rsi_14[i] <= 55:
                new_signal = BASE_SIZE * 0.6
            elif htf_bearish and 45 <= rsi_14[i] <= 70:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes very overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Exit short when RSI becomes very oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
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