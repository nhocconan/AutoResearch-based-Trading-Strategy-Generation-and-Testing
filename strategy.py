#!/usr/bin/env python3
"""
Experiment #007: 1d Primary + 1w HTF Trend + Volatility Filter + RSI Entry

Hypothesis: After 6 failed experiments with complex regime switching, try SIMPLER approach:
1. 1w HMA(21) for MAJOR trend direction (call ONCE before loop via mtf_data)
2. 1d RSI(3) for entry timing (Connors-style fast RSI, proven on ETH)
3. Volatility filter: ATR(7)/ATR(30) ratio to avoid dead chop periods
4. BB(20,2.0) for mean-reversion entries in range markets
5. Simple 2.5*ATR stoploss

Why this should work:
- 1w HTF filter prevents counter-trend trades (major failure in 2022 crash)
- RSI(3) extremes catch short-term reversals within major trend
- Volatility filter avoids trading during dead periods (fee drag)
- 1d timeframe targets 10-30 trades/year (optimal for fee efficiency)
- Minimal complexity reduces overfitting risk (learned from 6 failures)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi3_1w_hma_vol_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_3 = calculate_rsi(close, 3)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # 1d HMA for local trend
    hma_1d_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(60, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1W HTF TREND BIAS ===
        # Price above 1w HMA = bullish bias, below = bearish
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        local_bullish = close[i] > hma_1d_21[i]
        local_bearish = close[i] < hma_1d_21[i]
        
        # === VOLATILITY FILTER ===
        # ATR(7)/ATR(30) ratio - avoid dead chop (<0.6) or extreme vol (>2.0)
        vol_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 1.0
        vol_ok = 0.6 < vol_ratio < 2.0
        
        # === RSI(3) EXTREMES FOR ENTRY ===
        # Long: RSI(3) < 15 (oversold pullback in uptrend)
        # Short: RSI(3) > 85 (overbought pullback in downtrend)
        rsi3_oversold = rsi_3[i] < 15
        rsi3_overbought = rsi_3[i] > 85
        
        # === BOLLINGER BAND POSITION ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        bb_mid_cross_up = close[i] > bb_mid[i] and close[i-1] <= bb_mid[i] if i > 0 else False
        bb_mid_cross_down = close[i] < bb_mid[i] and close[i-1] >= bb_mid[i] if i > 0 else False
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_bullish or local_bullish:
            current_size = BASE_SIZE
        elif htf_bearish and local_bearish:
            current_size = STRONG_SIZE
        elif htf_bearish or local_bearish:
            current_size = BASE_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1w bullish + RSI(3) oversold OR BB lower break + vol filter
        if htf_bullish and vol_ok:
            if rsi3_oversold or (bb_oversold and rsi_3[i] < 30):
                new_signal = current_size
        
        # SHORT ENTRY: 1w bearish + RSI(3) overbought OR BB upper break + vol filter
        elif htf_bearish and vol_ok:
            if rsi3_overbought or (bb_overbought and rsi_3[i] > 70):
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), allow weaker entry
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_3[i] < 25:
                new_signal = current_size * 0.8
            elif htf_bearish and rsi_3[i] > 75:
                new_signal = -current_size * 0.8
        
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
            # Exit long if 1w trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1w trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI MEAN REVERSION EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI(3) becomes overbought (mean reversion)
            if position_side > 0 and rsi_3[i] > 75:
                rsi_exit = True
            # Exit short when RSI(3) becomes oversold (mean reversion)
            if position_side < 0 and rsi_3[i] < 25:
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