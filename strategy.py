#!/usr/bin/env python3
"""
Experiment #273: 1d Primary + 1w HTF — Simplified Trend + Mean Reversion Hybrid

Hypothesis: After 246 failed experiments with complex regime filters, simplify for 1d:
1. 1w HMA(21) for PRIMARY trend (ultra-slow, minimal whipsaw on daily)
2. 1d RSI(14) for entry timing (30/70 thresholds for mean reversion in trend)
3. 1d ROC(10) momentum confirmation (avoids catching falling knives)
4. ATR(14) trailing stoploss at 3.0 ATR (wider stops for daily noise)
5. MINIMAL filters — just trend direction + RSI extreme (ensures trades generate)

Key insight from failures: Complex regime filters (Chop+ADX+multiple) = 0 trades.
1d timeframe naturally filters noise. Don't over-engineer.

Position sizing: 0.25 base, 0.30 strong (conservative for daily volatility)
Target: 20-50 trades/year (80-200 over 4 years train)
Stoploss: 3.0 * ATR trailing (daily needs wider stops)

Why this might work:
- 1w trend filter eliminates counter-trend trades in strong moves
- RSI 30/70 on daily catches pullbacks in trend (not extremes like 10/90)
- ROC momentum ensures we're not entering dead markets
- Simple logic = more trades = better statistical significance
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_roc_1w_v1"
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

def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum)."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    roc = roc.fillna(0).values
    return roc

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    roc_10 = calculate_roc(close, 10)
    sma_50 = calculate_sma(close, 50)
    hma_1d_21 = calculate_hma(close, 21)
    
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
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(roc_10[i]):
            continue
        
        # === 1W TREND REGIME (primary direction filter) ===
        # Bull: price above 1w HMA (major uptrend)
        # Bear: price below 1w HMA (major downtrend)
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        price_above_hma = close[i] > hma_1d_21[i]
        price_below_hma = close[i] < hma_1d_21[i]
        price_above_sma = close[i] > sma_50[i]
        price_below_sma = close[i] < sma_50[i]
        
        # === MOMENTUM FILTER ===
        momentum_positive = roc_10[i] > 0.0
        momentum_negative = roc_10[i] < 0.0
        momentum_strong_pos = roc_10[i] > 2.0
        momentum_strong_neg = roc_10[i] < -2.0
        
        # === RSI THRESHOLDS (relaxed for more trades on daily) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        rsi_neutral = 45.0 < rsi_14[i] < 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (trend-aligned mean reversion)
        # Primary: Bull regime + RSI pullback + momentum turning positive
        if regime_bull and rsi_oversold and momentum_positive:
            new_signal = BASE_SIZE
        
        # Strong long: Bull regime + extreme RSI + price above 1d HMA
        if regime_bull and rsi_extreme_oversold and price_above_hma:
            new_signal = STRONG_SIZE
        
        # Trend continuation: Bull regime + price above HMA + RSI neutral-bull
        if regime_bull and price_above_hma and 45 < rsi_14[i] < 60:
            if new_signal == 0.0:
                new_signal = BASE_SIZE * 0.8
        
        # SHORT ENTRIES (trend-aligned mean reversion)
        # Primary: Bear regime + RSI rally + momentum turning negative
        if regime_bear and rsi_overbought and momentum_negative:
            if new_signal == 0.0 or abs(new_signal) < BASE_SIZE:
                new_signal = -BASE_SIZE
        
        # Strong short: Bear regime + extreme RSI + price below 1d HMA
        if regime_bear and rsi_extreme_overbought and price_below_hma:
            if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                new_signal = -STRONG_SIZE
        
        # Trend continuation: Bear regime + price below HMA + RSI neutral-bear
        if regime_bear and price_below_hma and 40 < rsi_14[i] < 55:
            if new_signal == 0.0:
                new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~15 days on 1d)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 35 and price_above_hma:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and rsi_14[i] < 65 and price_below_hma:
                new_signal = -BASE_SIZE * 0.7
            elif regime_bull and momentum_strong_pos:
                new_signal = BASE_SIZE * 0.6
            elif regime_bear and momentum_strong_neg:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_sma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_sma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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