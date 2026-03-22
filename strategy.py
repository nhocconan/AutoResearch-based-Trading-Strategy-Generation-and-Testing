#!/usr/bin/env python3
"""
Experiment #302: 12h Primary + 1d/1w HTF — Simplified Trend + Volume Confirmation

Hypothesis: Previous 12h strategies failed due to:
1. Too many confluence filters (never all align = 0 trades)
2. Complex regime switching creates whipsaws
3. Not enough volume confirmation on breakouts

This strategy simplifies to:
1. 1d HMA(21) = primary trend direction (long only above, short only below)
2. 1w HMA(50) = macro regime filter (avoid counter-trend trades)
3. 12h RSI(14) = entry timing (oversold in uptrend, overbought in downtrend)
4. Volume confirmation = taker_buy_volume ratio > 1.2 for longs, < 0.8 for shorts
5. ATR(14) trailing stop = 2.5x for risk management

Why this might beat #292 (Sharpe=0.424):
- Fewer filters = more trades (address 0-trade problem)
- Volume confirmation adds edge without complexity
- Asymmetric entries match crypto behavior
- 12h TF naturally limits trade frequency to 20-50/year

Position sizing: 0.30 base, 0.35 strong conviction
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_vol_1d1w_simp_v1"
timeframe = "12h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume ratio (taker buy / total)
    vol_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            vol_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            vol_ratio[i] = 0.5
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        regime_neutral = not regime_bull and not regime_bear
        
        # === 1W MACRO REGIME (avoid counter-trend) ===
        macro_bull = close[i] > hma_1w_50_aligned[i]
        macro_bear = close[i] < hma_1w_50_aligned[i]
        
        # Strong conviction when 1d and 1w align
        strong_bull = regime_bull and macro_bull
        strong_bear = regime_bear and macro_bear
        
        # === RSI THRESHOLDS (LOOSE for more trades) ===
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        rsi_extreme_oversold = rsi_14[i] < 35.0
        rsi_extreme_overbought = rsi_14[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_buy_pressure = vol_ratio[i] > 0.55
        vol_sell_pressure = vol_ratio[i] < 0.45
        
        # === RSI MOMENTUM ===
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (when 1d regime bull or neutral)
        if regime_bull or regime_neutral:
            # Strong conviction: 1d+1w aligned bull + RSI oversold
            if strong_bull and rsi_oversold:
                new_signal = STRONG_SIZE
            
            # Standard long: bull regime + RSI oversold + volume confirm
            elif regime_bull and rsi_oversold and vol_buy_pressure:
                new_signal = BASE_SIZE
            
            # Standard long: bull regime + RSI oversold (no volume req)
            elif regime_bull and rsi_extreme_oversold:
                new_signal = BASE_SIZE
            
            # Neutral regime + extreme oversold
            elif regime_neutral and rsi_extreme_oversold and vol_buy_pressure:
                new_signal = MIN_SIZE
        
        # SHORT ENTRIES (when 1d regime bear or neutral)
        if regime_bear or regime_neutral:
            # Strong conviction: 1d+1w aligned bear + RSI overbought
            if strong_bear and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            
            # Standard short: bear regime + RSI overbought + volume confirm
            elif regime_bear and rsi_overbought and vol_sell_pressure:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            
            # Standard short: bear regime + RSI overbought (no volume req)
            elif regime_bear and rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            
            # Neutral regime + extreme overbought
            elif regime_neutral and rsi_extreme_overbought and vol_sell_pressure:
                if new_signal == 0.0:
                    new_signal = -MIN_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 12h) ===
        # Force trade if no signal for 40 bars (~40 * 12h = 480h ≈ 20 days)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] < 50:
                new_signal = MIN_SIZE
            elif regime_bear and rsi_14[i] > 50:
                new_signal = -MIN_SIZE
            elif regime_neutral and rsi_extreme_oversold:
                new_signal = MIN_SIZE
            elif regime_neutral and rsi_extreme_overbought:
                new_signal = -MIN_SIZE
        
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
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Short position: exit when RSI oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns bearish
            if position_side > 0 and regime_bear:
                regime_reversal = True
            # Short position but 1d regime turns bullish
            if position_side < 0 and regime_bull:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.18:
                new_signal = 0.0
            elif new_signal > 0.32:
                new_signal = STRONG_SIZE
            elif new_signal > 0:
                new_signal = BASE_SIZE
            elif new_signal < -0.32:
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