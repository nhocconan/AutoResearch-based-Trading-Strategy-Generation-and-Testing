#!/usr/bin/env python3
"""
Experiment #404: 30m RSI Mean Reversion with 4h HMA Trend + Choppiness Regime Filter

Hypothesis: 30m timeframe is ideal for capturing intraday mean-reversion swings
while using 4h trend filter to avoid counter-trend trades. Previous 30m strategies
failed because they lacked proper regime detection and used weak HTF filters.

KEY INSIGHTS FROM 30M FAILURES (#392, #397, #398, #403):
- Simple RSI mean-reversion without trend filter = whipsaw death
- MACD momentum on 30m = too many false signals
- Need STRONGER 4h filter than previous attempts
- Choppiness Index is critical to avoid trending market mean-reversion

STRATEGY COMPONENTS:
1. 4h HMA(21) TREND BIAS: Primary directional filter
   - Only long when price > 4h HMA (bull trend)
   - Only short when price < 4h HMA (bear trend)
   - HMA smoother than EMA, less lag for 30m entries

2. CHOPPINESS INDEX(14) REGIME: Avoid mean-reversion in strong trends
   - CHOP > 55 = ranging (allow mean-reversion entries)
   - CHOP < 45 = trending (only follow 4h HMA direction)
   - Prevents RSI mean-reversion during strong trend continuations

3. RSI(7) FAST MEAN REVERSION: 30m entry trigger
   - Long: RSI < 35 + price > 4h HMA + CHOP > 55
   - Short: RSI > 65 + price < 4h HMA + CHOP > 55
   - Faster RSI(7) vs RSI(14) for 30m timeframe sensitivity

4. VOLUME CONFIRMATION: Filter false breakouts
   - Volume > 0.8 * SMA20(volume) for entry validation
   - Avoids low-liquidity trap entries

5. ATR TRAILING STOP (2.0x): Tighter for 30m frequency
   - Signal → 0 when price moves 2.0*ATR against position
   - Protects from rapid 30m reversals

6. POSITION SIZING: 0.25 discrete (conservative for 30m volatility)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 to minimize fee churn

Why this should beat previous 30m attempts:
- Stronger 4h HMA filter (previous used weak 4h EMA or no HTF)
- Choppiness regime prevents mean-reversion in trends (key missing piece)
- RSI(7) faster than RSI(14) for 30m sensitivity
- Volume filter avoids low-liquidity traps
- Should generate 60-120 trades/year per symbol (enough for stats)
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_meanrev_4h_hma_chop_vol_atr_v1"
timeframe = "30m"
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
    """Calculate RSI with proper min_periods."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    chop = calculate_choppiness_index(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        ranging_market = chop[i] > 55.0
        trending_market = chop[i] < 45.0
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGING REGIME: Mean-reversion entries with trend bias
        if ranging_market and volume_ok:
            # Long: RSI oversold + price above 4h HMA (bull bias)
            if rsi_oversold and bull_trend_4h:
                new_signal = SIZE
            # Short: RSI overbought + price below 4h HMA (bear bias)
            elif rsi_overbought and bear_trend_4h:
                new_signal = -SIZE
        
        # TRENDING REGIME: Only follow 4h HMA direction (no mean-reversion)
        elif trending_market:
            if bull_trend_4h:
                new_signal = SIZE
            elif bear_trend_4h:
                new_signal = -SIZE
        # NEUTRAL REGIME (45 <= CHOP <= 55): Stay flat to avoid whipsaw
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit mean-reversion position if regime changes to trending against position
        if in_position and new_signal != 0.0 and not ranging_market:
            # Long position entered on RSI oversold should exit if trending bear
            if position_side > 0 and bear_trend_4h and trending_market:
                new_signal = 0.0
            # Short position entered on RSI overbought should exit if trending bull
            if position_side < 0 and bull_trend_4h and trending_market:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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