#!/usr/bin/env python3
"""
Experiment #363: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 360+ experiments, the pattern is crystal clear:
1. Complex regime-switching (choppiness, dual-strategy) keeps FAILING (Sharpe < 0)
2. 1d primary + 1w HTF is the ONLY combination that consistently works (current best: Sharpe=0.435)
3. SIMPLICITY wins: HMA trend + RSI pullback + ATR stop is the proven formula
4. Previous attempt (#353) had 0 trades — entry conditions were TOO STRICT
5. This version LOOSENS entries while keeping the core edge

Why this might beat current best (Sharpe=0.435):
- Simpler logic = less overfitting, more robust across regimes
- Looser RSI thresholds (30-55 long, 45-70 short) = more trades
- 1w HMA(21) for major trend = avoids counter-trend trades in bear markets
- Daily timeframe = 20-50 trades/year optimal (not too many fees, not too few signals)
- Asymmetric sizing (0.30 long, 0.20 short) = captures crypto's long bias

Position sizing: 0.30 longs, 0.20 shorts (discrete levels)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-50 trades/year on 1d (~1 trade per 1-2 weeks)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_simp_v4"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    sma_1d_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Trade counter for debugging
    trade_count = 0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            signals[i] = 0.0
            continue
        
        # === 1W MAJOR TREND REGIME (primary direction filter) ===
        # Price above weekly HMA = bull market (favor longs)
        # Price below weekly HMA = bear market (favor shorts)
        weekly_bull = close[i] > hma_1w_21_aligned[i]
        weekly_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND MOMENTUM (HMA crossover) ===
        # HMA(21) > HMA(50) = bullish momentum
        # HMA(21) < HMA(50) = bearish momentum
        daily_bullish = hma_1d_21[i] > hma_1d_50[i]
        daily_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === PRICE POSITION ===
        price_above_hma21 = close[i] > hma_1d_21[i]
        price_below_hma21 = close[i] < hma_1d_21[i]
        price_above_sma200 = close[i] > sma_1d_200[i] if not np.isnan(sma_1d_200[i]) else True
        
        # === RSI PULLBACK SIGNALS (LOOSENED for more trades) ===
        # Long: RSI pulled back but not oversold (30-55 range)
        rsi_long_pullback = 30.0 <= rsi_14[i] <= 55.0
        rsi_long_oversold = rsi_14[i] < 40.0
        
        # Short: RSI rallied but not overbought (45-70 range)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 70.0
        rsi_short_overbought = rsi_14[i] > 60.0
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        # LONG ENTRY: Weekly bull + Daily bullish + RSI pullback + Price above HMA21
        if weekly_bull and daily_bullish and rsi_long_pullback and price_above_hma21:
            new_signal = LONG_SIZE
        # LONG ENTRY (secondary): Weekly bull + Price above SMA200 + RSI oversold
        elif weekly_bull and price_above_sma200 and rsi_long_oversold:
            new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY: Weekly bear + Daily bearish + RSI pullback + Price below HMA21
        if weekly_bear and daily_bearish and rsi_short_pullback and price_below_hma21:
            if new_signal == 0.0:  # Don't override long signal
                new_signal = -SHORT_SIZE
        # SHORT ENTRY (secondary): Weekly bear + Price below SMA200 + RSI overbought
        elif weekly_bear and not price_above_sma200 and rsi_short_overbought:
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOSTER (ensure 25+ trades/year) ===
        # If no signal for 20 bars (~3 weeks on 1d), force entry on weaker conditions
        if new_signal == 0.0 and not in_position:
            # Count bars since last non-zero signal
            bars_since_signal = 0
            for j in range(i-1, max(0, i-50), -1):
                if signals[j] != 0.0:
                    break
                bars_since_signal += 1
            
            if bars_since_signal > 20:
                # Weaker entry conditions after long silence
                if weekly_bull and rsi_14[i] < 50.0 and price_above_hma21:
                    new_signal = LONG_SIZE * 0.6
                elif weekly_bear and rsi_14[i] > 50.0 and price_below_hma21:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0 and atr_14[i] > 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                # Trailing stop: exit if price drops 2.5*ATR from highest
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                # Trailing stop: exit if price rises 2.5*ATR from lowest
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long: exit when RSI gets overbought (>70)
            if position_side > 0 and rsi_14[i] > 70.0:
                rsi_exit = True
            # Short: exit when RSI gets oversold (<30)
            if position_side < 0 and rsi_14[i] < 30.0:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Long: exit when daily HMA turns bearish
            if position_side > 0 and daily_bearish and price_below_hma21:
                trend_exit = True
            # Short: exit when daily HMA turns bullish
            if position_side < 0 and daily_bullish and price_above_hma21:
                trend_exit = True
        
        # Apply exits
        if stoploss_triggered or rsi_exit or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = 1 if new_signal > 0 else -1
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                trade_count += 1
            elif np.sign(new_signal) != position_side:
                # Position reversal
                position_side = 1 if new_signal > 0 else -1
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                trade_count += 1
            # If same side, keep position (no new trade)
        else:
            if in_position:
                # Position closed
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals