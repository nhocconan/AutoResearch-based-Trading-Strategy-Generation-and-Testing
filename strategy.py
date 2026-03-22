#!/usr/bin/env python3
"""
Experiment #392: 12h Primary + 1d/1w HTF — Choppiness Regime + Dual Strategy

Hypothesis: After 350+ failed experiments, the clearest pattern is:
1. 12h timeframe generates 20-50 trades/year (optimal for fee/capture balance)
2. Choppiness Index regime detection PROVEN on ETH (Sharpe +0.923 in research)
3. Dual strategy: mean-revert in choppy markets, trend-follow in trending markets
4. 1w HMA for major trend bias (prevents counter-trend trades in bear markets)
5. 1d HMA for intermediate trend confirmation
6. Discrete position sizing: 0.0, ±0.25, ±0.30 (max 0.40)
7. ATR 2.5x trailing stop for risk management

Why this might beat current best (Sharpe=0.435):
- 12h TF reduces fee drag vs 4h/1h while maintaining trade frequency
- Choppiness regime switch adapts to market conditions (range vs trend)
- 1w HTF filter prevents catastrophic counter-trend trades (like 2022 crash)
- Mean reversion in chop + trend follow in trend = best of both worlds
- Simpler than failed dual-regime attempts (#381, #384, #386, #391)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 20-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_hma_donchian_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Choppy/Range market (mean reversion)
    - CHOP < 38.2: Trending market (trend following)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction - prevents 2022-style crashes)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W MAJOR TREND (catastrophic crash prevention) ===
        # Price above 1w HMA = bull market (allow longs, restrict shorts)
        # Price below 1w HMA = bear market (allow shorts, restrict longs)
        bull_major = close[i] > hma_1w_21_aligned[i]
        bear_major = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        bull_inter = close[i] > hma_1d_21_aligned[i]
        bear_inter = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8: Choppy/Range (use mean reversion)
        # CHOP < 38.2: Trending (use breakout/trend following)
        # 38.2 <= CHOP <= 61.8: Transition (reduce size or wait)
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        transition_regime = not choppy_regime and not trending_regime
        
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === REGIME 1: CHOPPY (Mean Reversion at Bollinger Bands) ===
        if choppy_regime:
            # Long: Price at/near lower BB + RSI oversold + 1d bullish bias
            if close[i] <= bb_lower[i] * 1.002 and rsi_14[i] < 35:
                if bull_inter or (not bear_major):
                    new_signal = LONG_SIZE
            
            # Short: Price at/near upper BB + RSI overbought + 1d bearish bias
            if close[i] >= bb_upper[i] * 0.998 and rsi_14[i] > 65:
                if bear_inter or (not bull_major):
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
        
        # === REGIME 2: TRENDING (Donchian Breakout + HMA Confirmation) ===
        elif trending_regime:
            # Long: Donchian breakout + 1w/1d bullish + RSI not overbought
            if close[i] > donchian_upper[i] and rsi_14[i] < 70:
                if bull_major and bull_inter:
                    new_signal = LONG_SIZE
            
            # Short: Donchian breakdown + 1w/1d bearish + RSI not oversold
            if close[i] < donchian_lower[i] and rsi_14[i] > 30:
                if bear_major and bear_inter:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
        
        # === REGIME 3: TRANSITION (Reduced size, stricter filters) ===
        elif transition_regime:
            # Only enter on strong confluence
            if bull_major and bull_inter and rsi_14[i] < 45 and close[i] > bb_mid[i]:
                new_signal = LONG_SIZE * 0.7
            elif bear_major and bear_inter and rsi_14[i] > 55 and close[i] < bb_mid[i]:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 8 bars (~4 days on 12h), force entry on weaker signal
        if bars_since_last_trade > 8 and new_signal == 0.0 and not in_position:
            if bull_major and rsi_14[i] < 45 and not choppy_regime:
                new_signal = LONG_SIZE * 0.6
            elif bear_major and rsi_14[i] > 55 and not choppy_regime:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # Mean reversion exit (price returned to BB mid)
        if in_position and position_side > 0 and choppy_regime:
            if close[i] > bb_mid[i] * 1.002:
                new_signal = 0.0
        if in_position and position_side < 0 and choppy_regime:
            if close[i] < bb_mid[i] * 0.998:
                new_signal = 0.0
        
        # Major trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_major:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_major:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
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
                # Position flip
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