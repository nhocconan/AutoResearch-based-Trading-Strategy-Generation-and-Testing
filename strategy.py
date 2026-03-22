#!/usr/bin/env python3
"""
Experiment #475: 15m Mean-Reversion with 4h HMA Trend Bias

Hypothesis: 15m timeframe is ideal for mean-reversion strategies that align with 
higher timeframe trend. After 474 experiments, key insight is that BTC/ETH need
ASYMMETRIC logic: buy dips in uptrend, short rallies in downtrend.

Strategy Logic:
1. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Bull: price > 4h HMA → favor long entries on RSI dips
   - Bear: price < 4h HMA → favor short entries on RSI rallies

2. 15m RSI(7) FAST MEAN-REVERSION:
   - Long: RSI < 35 (oversold dip in uptrend)
   - Short: RSI > 65 (overbought rally in downtrend)
   - Fast RSI(7) catches quick reversals on 15m

3. SMA(200) REGIME CONFIRMATION:
   - Price > SMA200 = bull market (prioritize longs)
   - Price < SMA200 = bear market (prioritize shorts)

4. VOLUME CONFIRMATION:
   - Volume > 1.1 * Volume_SMA(20) confirms genuine moves
   - Filters low-liquidity false signals

5. ATR(14) TRAILING STOP at 2.5x:
   - Tight enough for 15m, wide enough to avoid noise
   - Signal → 0 when stop hit

6. POSITION SIZING: 0.25 discrete
   - Conservative for 15m volatility
   - Discrete levels minimize fee churn

Why 15m works:
- Fast enough to catch intraday mean-reversion
- 4h HMA provides stable trend filter without whipsaw
- Loose RSI thresholds (35/65) ensure sufficient trades
- Should generate 50-100 trades/year per symbol

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_meanrev_4h_hma_rsi7_sma200_vol_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index with fast period for 15m."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period=20):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Fast RSI for 15m mean-reversion
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_ema(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        if vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === SMA200 REGIME ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.1 * vol_sma[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35  # Long entry threshold
        rsi_overbought = rsi[i] > 65  # Short entry threshold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Buy dips (mean-reversion long)
        if bull_trend and above_sma200:
            if rsi_oversold and vol_confirmed:
                new_signal = SIZE
        
        # BEAR REGIME: Short rallies (mean-reversion short)
        if bear_trend and below_sma200:
            if rsi_overbought and vol_confirmed:
                new_signal = -SIZE
        
        # TRANSITION/NEUTRAL: Both signals allowed with stricter RSI
        if not bull_trend or not above_sma200:
            if rsi[i] < 28 and vol_confirmed:  # Deeper oversold
                new_signal = SIZE
        
        if not bear_trend or not below_sma200:
            if rsi[i] > 72 and vol_confirmed:  # Deeper overbought
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals